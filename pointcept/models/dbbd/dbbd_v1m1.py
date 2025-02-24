"""
Pretraining TODO

Author: Anthony Yaghi, Manuel Philipp Vogel
"""
from typing import List, Dict, Any
from itertools import chain
import torch.distributed as dist
import numpy as np
import random
import torch
torch.manual_seed(12)
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.dbbd.Aggregator import MaxPoolAggregator
from pointcept.models.dbbd.Propagator import ConcatPropagation

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.utils.comm import get_world_size
import pointops
from torch_geometric.nn.pool import voxel_grid
from timm.models.layers import trunc_normal_

def inference(encoder, points_tensor, view_data_dict=None):
    # Encode the points using the dynamic encoder
    device = points_tensor.device
    resized_points_tensor = points_tensor.reshape(points_tensor.shape[0] *  points_tensor.shape[1], points_tensor.shape[2]) # (B, N, D) -> (B*N, D)
    
    offset_arr = []
    for i in range(points_tensor.shape[0]):
        offset_arr.append((i+1)*points_tensor.shape[1]) # Each offset is the number of points in the previous batch
    offset_arr = torch.tensor(offset_arr,device=device)


    # points_dict = {"feat": F.pad(resized_points_tensor, padding_size), "coord": resized_points_tensor[:, :3], "grid_size": 0.01,
    #                "offset": offset_arr}

    #point_dict issue for sparseconv (New encoder)
    points_dict = {"feat": view_data_dict["feat"], "coord": resized_points_tensor[:, :3], "grid_coord": view_data_dict['grid_coord'], 
                   "offset": offset_arr}
    
    # NOTE Masked variables added NOTE #
    points_masked_dict = {"feat": view_data_dict["masked_feat"],"coord": resized_points_tensor[:, :3], "grid_coord": view_data_dict['grid_coord'], 
                   "offset": offset_arr}
    
    # point_features = encoder(points_dict)["feat"]
    
    # shape: [20000, 96] [B*N, output_dim]
    point_features = encoder(points_dict) # (B*N, output_dim)
    
    point_masked_features = encoder(points_masked_dict) # NOTE Masked variables added NOTE #
    
    # point_features = F.pad(point_features, (0, 128 - 64))  # since the output is 64, pad to 128

    # shape: [20000] [B*N]
    batch = offset2batch(offset_arr)
    
    # shape: [4] [B]
    # tensor([5000, 5000, 5000, 5000]) [N, N, N, N]
    # shape: [B, N]
    batch_count = batch.bincount()
    point_features_split = point_features.split(list(batch_count))
    point_features_split = torch.stack(point_features_split)
    
    # NOTE Masked variables added NOTE #
    point_masked_features_split = point_masked_features.split(list(batch_count))
    point_masked_features_split = torch.stack(point_masked_features_split)
    
    # shape: [4, 5000, 96] [B, N, output_dim]
    return point_features_split, point_masked_features_split


def encode_and_propagate(region: List[Dict[str, Any]], # (levelB, ...)
                         encoder, 
                         aggregator,
                         propagation_method, 
                         view_data_dict,
                         parent_feature: List[torch.Tensor] = None, # (levelB, ) if there's a parent feature list, 
                         level: int = 0,
                         output_dim=128) -> List[Dict[str, Any]]:
    
    
    # Iterate through regions and get corresponding indices then points from transformed points -> List of points vectors (levelB, levelN, D)
    points_tensor_list = []
    
    for i, reg in enumerate(region):
        batch_idx = reg['batch_idx']
        
        # shape: [5000, 3] [N, D]
        corresponding_transformed_points = view_data_dict["coord"][batch_idx]
        # shape: [5000] [N]
        indices = np.array(reg['points_indices'], dtype=int) # (levelN,)
  
        if len(indices) == 0:
            raise Exception("Got a region with no indics")
        else:
            # shape: [5000, 3] [N, D]
            points_tensor = corresponding_transformed_points[indices] # (levelN, D)
            # shape: [5000, 96] [N, output_dim]
            points_tensor = F.pad(points_tensor, (0, 99 - 2*points_tensor.shape[1]))

            # Propagate if there's a parent (Get the parent superpoints from the hierarchy for each region in the list)
            if parent_feature is not None: # parent_feature: shape = [8, 96] [2xB, output_dim]
                # shape: [500, 96] [levelN, output_dim]
                points_tensor = propagation_method.propagate(parent_feature[i], points_tensor) # (levelN, output_dim)
            
            # shape: [4 x [5000, 96]] [B x [N, output_dim]]
            points_tensor_list.append(points_tensor)

    # shape: [4, 5000, 96] [B, N, output_dim]
    batched_tensor = torch.stack(points_tensor_list) # (levelB, levelN, D or output_dim) # Assuming all regions on a level have the same number of points

    # Encode
    # if batched_tensor.shape[2] == 6:
    #     padding_size = (0, output_dim)  # pad the last dimension (0 elems in front, 128 after)
    # elif batched_tensor.shape[2] ==output_dim:
    #     padding_size = (0, 6)
    # else:
    #     print(f"PROBLEM WITH TENSOR SIZE {batched_tensor.shape}")
    
    # NOTE Masked variables added NOTE #
    # shape: [4, 5000, 96] [B, N, output_dim]
    batched_point_features, batched_point_masked_features = inference(encoder, batched_tensor, view_data_dict)

    # Aggregate
    # shape: [4, 96] [B, output_dim]
    batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)

    # Iterate through list of regions and set as the corresponding superpoints for the level
    parent_feature_list = []
    next_level_sub_regions = []
    for i, reg in enumerate(region):
        # reg is a dict: {
        # points, points_indices, sub_regions, batch_idx, super_point_branch1, super_point1, level_branch1    
        # }
        reg['super_point_branch1'] = batched_region_feature[i] # (output_dim,)
        reg['super_point1'] = batched_point_features[i] # (N, output_dim)
        # NOTE Masked variables added NOTE #
        reg['super_point_masked1'] = batched_point_masked_features[i] # (N, output_dim)
        reg['level_branch1'] = level 

        # Duplicate in parent_feature array based on number of upcoming subregions
        if len(reg['sub_regions']) <= 0:
            continue
            
        for sub_region in reg['sub_regions']:
            if len(sub_region['points_indices']) > 0:
                parent_feature_list.append(batched_region_feature[i])
                next_level_sub_regions.append(sub_region)
    if len(next_level_sub_regions) > 0 and len(parent_feature_list)>0:
        assert len(next_level_sub_regions) == len(parent_feature_list), "Mismatch between next level subregions and parent features list"
        encode_and_propagate(next_level_sub_regions, encoder, aggregator, propagation_method, view_data_dict, parent_feature=parent_feature_list, level=level+1)
    
    return region

def encode_and_aggregate(region: List[Dict[str, Any]], # (levelB, ...) 
                         encoder, 
                         aggregator,
                         view_data_dict, # (B, N0, 3)
                         level: int = 0,
                         max_levels: int = 1,
                         output_dim=128) -> Dict[str, Any]:
    
    if level!=max_levels:
        previous_level_sub_regions = [] # (levlB, ...)
        for reg in region:
            if reg['sub_regions']:
                for sub_region in reg['sub_regions']:
                    if len(sub_region['points_indices']) > 0:
                        previous_level_sub_regions.append(sub_region)

        encode_and_aggregate(previous_level_sub_regions, encoder, aggregator, view_data_dict, level=level+1, max_levels= max_levels)

        super_points_from_previous_level = []
        for reg in region:
            if reg['sub_regions']:
                for sub_region in reg['sub_regions']:
                    super_points_from_previous_level.append(sub_region['super_point_branch2'])
        

        # shape: [8, 96] [B * num_sample_lvl, output_dim]
        batched_tensor = torch.stack(super_points_from_previous_level) # (levelB, C)
        # shape: [8, 1, 96] [B * num_sample_lvl, 1, output_dim]
        batched_tensor = batched_tensor.unsqueeze(1) # (levelB, 1, C)

        # if batched_tensor.shape[2] == 6:
        #     padding_size = (0, output_dim)  # pad the last dimension (0 elems in front, 128 after)
        # elif batched_tensor.shape[2] ==output_dim:
        #     padding_size = (0, 6)
        # else:
        #     print(f"PROBLEM WITH TENSOR SIZE {batched_tensor.shape}")

        # shape: [8, 1, 96] [B * num_sample_lvl, 1, output_dim]
        batched_point_features, batched_point_masked_features = inference(encoder, batched_tensor, view_data_dict)

        # Aggregate
        # shape [8, 96] [B * num_sample_lvl, output_dim]
        batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)

        # NOTE and TODO: Why are we not also adding super_point2 here ?
        for i, reg in enumerate(region):
            reg['super_point_branch2'] = batched_region_feature[i] # (output_dim,)
            reg['level_branch2'] = level
            
            reg['super_point2'] = batched_point_features[i] # NOTE: Added by Angelo for testing
            reg['super_point_masked2'] = batched_point_masked_features[i] # NOTE: Added by Angelo for testing
            
    else:
        # IF LAST LEVEL
        points_tensor_list = []
        for i, reg in enumerate(region):
            batch_idx = reg['batch_idx']
            
            # shape: [5000, 3] [N, D]
            corresponding_transformed_points = view_data_dict["coord"][batch_idx]
            
            # shape: [5000] [N]
            indices = np.array(reg['points_indices'], dtype=int) # (levelN,)
    
            if len(indices) == 0:
                raise Exception("Got a region with no indics")
            else:
                # shape: [5000, 3] [N, D]
                points_tensor = corresponding_transformed_points[indices] # (levelN, D)
                # shape: [5000, 96] [N, output_dim]
                points_tensor = F.pad(points_tensor, (0, 99 - 2*points_tensor.shape[1]))
                
                # shape: [4 x [500, 96]] [B x [N, output_dim]]
                points_tensor_list.append(points_tensor)

        # shape: [4, 500, 96] [B, N, output_dim]
        batched_tensor = torch.stack(points_tensor_list) # (levelB, levelN, D or output_dim) # Assuming all regions on a level have the same number of points


        # # Encode
        # if batched_tensor.shape[2] == 6:
        #     padding_size = (0, output_dim)  # pad the last dimension (0 elems in front, 128 after)
        # elif batched_tensor.shape[2] ==output_dim:
        #     padding_size = (0, 6)
        # else:
        #     print(f"PROBLEM WITH TENSOR SIZE {batched_tensor.shape}")

        # shape: [4, 500, 96] [B, N, output_dim]
        batched_point_features, batched_point_masked_features = inference(encoder, batched_tensor, view_data_dict)
        
        # Aggregate
        # shape: [4, 96] [B, output_dim]
        batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)

        for i, reg in enumerate(region):
            reg['super_point_branch2'] = batched_region_feature[i] # (output_dim,)
            reg['super_point2'] = batched_point_features[i] # (N, output_dim)
            reg['super_point_masked2'] = batched_point_masked_features[i] # (N, output_dim)
            reg['level_branch2'] = level 

    return region

def collect_region_features_per_level(region: Dict[str, Any],
                                      features_dict_branch1: Dict[int, List[torch.Tensor]],
                                      features_dict_branch2: Dict[int, List[torch.Tensor]]) -> None:
    # Collect features from Branch 1
    if 'super_point_branch1' in region:
        level1 = region['level_branch1']
        if level1 not in features_dict_branch1:
            features_dict_branch1[level1] = []
        features_dict_branch1[level1].append(region['super_point_branch1'])

    # Collect features from Branch 2
    if 'super_point_branch2' in region:
        level2 = region['level_branch2']
        if level2 not in features_dict_branch2:
            features_dict_branch2[level2] = []
        features_dict_branch2[level2].append(region['super_point_branch2'])

    # Recursively collect from sub-regions
    for sub_region in region['sub_regions']:
        collect_region_features_per_level(sub_region, features_dict_branch1, features_dict_branch2)


def collect_region_features_per_points(region: Dict[str, Any],
                                      features_dict_branch1: Dict[int, List[torch.Tensor]],
                                      features_dict_branch2: Dict[int, List[torch.Tensor]]) -> None:
    # Collect features from Branch 1
    if 'super_point1' in region:
        # level1 (here 0 since mx_lvl=0) is the current branch1 level with respect to total max levels
        level1 = region['level_branch1']
        if level1 not in features_dict_branch1:
            features_dict_branch1[level1] = []
        # total desired shape: [1 x [5000, 96]] [mx_lvl x [N, output_dim]]
        features_dict_branch1[level1].append(region['super_point1']) # region['super_point1']: shape: [5000, 96] [N, output_dim]

    # Collect features from Branch 2
    if 'super_point2' in region:
        # level2 (here 0 since mx_lvl=0) is the current branch2 level with respect to total max levels
        level2 = region['level_branch2']
        if level2 not in features_dict_branch2:
            features_dict_branch2[level2] = []
        # total desired shape: [1 x [5000, 96]] [mx_lvl x [N, output_dim]]
        features_dict_branch2[level2].append(region['super_point2']) # NOTE region['super_point2']: shape: [1, 96] BUT WE WANT [N, output_dim]

    # Recursively collect from sub-regions
    for sub_region in region['sub_regions']:
        collect_region_features_per_points(sub_region, features_dict_branch1, features_dict_branch2)

# NOTE Masked variables added NOTE #  
def collect_region_masked_features_per_points(region: Dict[str, Any],
                                      masked_features_dict_branch1: Dict[int, List[torch.Tensor]],
                                      masked_features_dict_branch2: Dict[int, List[torch.Tensor]]) -> None:
    # Collect features from Branch 1
    if 'super_point_masked1' in region:
        # level1 (here 0 since mx_lvl=0) is the current branch1 level with respect to total max levels
        level1 = region['level_branch1']
        if level1 not in masked_features_dict_branch1:
            masked_features_dict_branch1[level1] = []
        # total desired shape: [1 x [5000, 96]] [mx_lvl x [N, output_dim]]
        masked_features_dict_branch1[level1].append(region['super_point_masked1'])

    # Collect features from Branch 2
    if 'super_point_masked2' in region:
        # level2 (here 0 since mx_lvl=0) is the current branch2 level with respect to total max levels
        level2 = region['level_branch2']
        if level2 not in masked_features_dict_branch2:
            masked_features_dict_branch2[level2] = []
        # total desired shape: [1 x [5000, 96]] [mx_lvl x [N, output_dim]]
        masked_features_dict_branch2[level2].append(region['super_point_masked2'])

    # Recursively collect from sub-regions
    for sub_region in region['sub_regions']:
        collect_region_masked_features_per_points(sub_region, masked_features_dict_branch1, masked_features_dict_branch2)

def compute_contrastive_loss_per_level(features_dict_branch1: Dict[int, List[torch.Tensor]],
                                       features_dict_branch2: Dict[int, List[torch.Tensor]],
                                       temperature: float = 0.07, device="cuda:0") -> torch.Tensor:
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for level in features_dict_branch1.keys():
        features_branch1 = features_dict_branch1[level]
        features_branch2 = features_dict_branch2.get(level, [])

        if not features_branch2:
            continue

        # Ensure the number of features is the same
        # print("LEVEL", level,  "a", len(features_branch1), "b", len(features_branch2))
        if len(features_branch1) != len(features_branch2):
            print(f"Mismatch at level {level}: {len(features_branch1)} vs {len(features_branch2)} features")
            continue

        features_branch1_tensor = torch.stack(features_branch1).to(device)
        features_branch2_tensor = torch.stack(features_branch2).to(device)

        # Normalize features
        features_branch1_tensor = F.normalize(features_branch1_tensor, dim=1)
        features_branch2_tensor = F.normalize(features_branch2_tensor, dim=1)

        # Compute logits
        logits = torch.mm(features_branch1_tensor, features_branch2_tensor.t()) / temperature
        labels = torch.arange(logits.size(0)).long().to(device)

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss

    return total_loss

def compute_contrastive_loss_per_points(features_dict_branch1: Dict[int, List[torch.Tensor]],
                                        features_dict_branch2: Dict[int, List[torch.Tensor]],
                                        temperature: float = 0.07, device="cuda") -> torch.Tensor:
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # features_dict_branch1: shape (1, [4, 5000, 96]) (mx_lvl, [B, N, output_dim])
    for level in features_dict_branch1.keys():
        # shape: [4, [5000, 96]] [B, [N, output_dim]]
        features_branch1 = features_dict_branch1[level]
        # desired shape: [4, [5000, 96]] [B, [N, output_dim]]
        features_branch2 = features_dict_branch2.get(level, [])

        # Skip if no corresponding features for the second branch
        if not features_branch2:
            continue

        if len(features_branch1) != len(features_branch2):
            print(f"Mismatch at level {level}: {len(features_branch1)} vs {len(features_branch2)} features")
            continue

        # Stack the features into tensors and move to the specified device
        # shape: [4, 5000, 96] [B, N, output_dim]
        features_branch1_tensor = torch.stack(features_branch1).to(device)
        # shape: [4, 5000, 96] [B, N, output_dim]
        features_branch2_tensor = torch.stack(features_branch2).to(device)

        # Normalize features along the feature dimension (dim=2)
        # shape: [4, 5000, 96] [B, N, output_dim]
        features_branch1_tensor = F.normalize(features_branch1_tensor, dim=2)
        features_branch2_tensor = F.normalize(features_branch2_tensor, dim=2)
        
        # Compute the logits (similarity between all pairs in the batch)
        # We use bmm for batch matrix multiplication
        # shape: [4, 5000, 5000] [B, N, N]
        logits = torch.bmm(features_branch1_tensor, features_branch2_tensor.transpose(1, 2)) / temperature

        # labels are the identity labels (diagonal elements are positive pairs)
        batch_size, seq_len, _ = logits.shape
        
        # shape: [5000] [N]
        # tensor([0, 1, 2, 3, ..., 4999])
        labels = torch.arange(seq_len).long().to(device)

        # Compute the loss per batch
        loss = criterion(logits.view(-1, seq_len), labels.repeat(batch_size).view(-1))
        
        total_loss += loss

    return total_loss

def combine_features(all_features_dict_branch: Dict, features_dict_branch: Dict):
    """
    Combine tensors from the current batch into an accumulated dictionary of features.

    :param all_features_dict_branch: Dictionary that accumulates tensors across batches.
                                     Keys represent dynamic levels, and values are lists of tensors.
    :param features_dict_branch: Dictionary containing tensors for the current batch.
                                 Keys represent dynamic levels, and values are lists of tensors.
    :return: None. The function updates all_features_dict_branch by appending the tensors
             from features_dict_branch at the corresponding levels.
    """
    for level, tensors in features_dict_branch.items():
        if level not in all_features_dict_branch:
            all_features_dict_branch[level] = []  # Initialize if the level doesn't exist
        all_features_dict_branch[level].extend(tensors)  # Append tensors

@MODELS.register_module("DBBD-v1m1")
class DBBD(nn.Module):
    def __init__(
        self,
        backbone,
        num_samples_per_level,
        max_levels,
        output_dim,
        device,
        loss_method,
        
        # NOTE Masked Variables added NOTE #
        backbone_in_channels,
        backbone_out_channels,
        mask_grid_size=0.1,
        mask_rate=0.4,
        view1_mix_prob=0,
        view2_mix_prob=0,
        matching_max_k=8,
        matching_max_radius=0.03,
        matching_max_pair=8192,
        nce_t=0.4,
        contrast_weight=1,
        reconstruct_weight=1,
        reconstruct_color=True,
        reconstruct_normal=True,

    ):
        super().__init__()
        self.point_encoder = build_model(backbone)
        self.aggregator = MaxPoolAggregator().to(device)
        self.propagation_method = ConcatPropagation().to(device)
        # self.propagation_method.update_feature_dim(input_dim=backbone["in_channels"], feature_dim=128)
        # NOTE Changed by Angelo for Testing NOTE #
        self.propagation_method.update_feature_dim(input_dim=99, feature_dim=96)
        self.output_dim = output_dim
        self.num_samples_per_level=num_samples_per_level
        self.max_levels=max_levels
        self.loss_method = loss_method
        
        # NOTE Masked Variables added NOTE #
        self.mask_grid_size = mask_grid_size
        self.mask_rate = mask_rate
        self.view1_mix_prob = view1_mix_prob
        self.view2_mix_prob = view2_mix_prob
        self.matching_max_k = matching_max_k
        self.matching_max_radius = matching_max_radius
        self.matching_max_pair = matching_max_pair
        self.nce_t = nce_t
        self.contrast_weight = contrast_weight
        self.reconstruct_weight = reconstruct_weight
        self.reconstruct_color = reconstruct_color
        self.reconstruct_normal = reconstruct_normal

        self.mask_token = nn.Parameter(torch.zeros(1, backbone_in_channels))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.color_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_color else None
        )
        self.normal_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_normal else None
        )
        self.nce_criteria = torch.nn.CrossEntropyLoss(reduction="mean")

    # NOTE Masked Functions added NOTE #
    @torch.no_grad()
    def generate_cross_masks(
        self, view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
    ):
        # union origin coord
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()
        view1_origin_coord_split = view1_origin_coord.split(list(view1_batch_count))
        view2_origin_coord_split = view2_origin_coord.split(list(view2_batch_count))
        union_origin_coord = torch.cat(
            list(
                chain.from_iterable(
                    zip(view1_origin_coord_split, view2_origin_coord_split)
                )
            )
        )
        union_offset = torch.cat(
            [view1_offset.unsqueeze(-1), view2_offset.unsqueeze(-1)], dim=-1
        ).sum(-1)
        union_batch = offset2batch(union_offset)

        # grid partition
        mask_patch_coord = union_origin_coord.div(self.mask_grid_size)
        mask_patch_grid_coord = torch.floor(mask_patch_coord)
        mask_patch_cluster = voxel_grid(
            pos=mask_patch_grid_coord, size=1, batch=union_batch, start=0
        )
        unique, cluster, counts = torch.unique(
            mask_patch_cluster, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        patch_max_point = counts.max().item()
        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
        patch2point_mask = torch.lt(
            torch.arange(patch_max_point).cuda().unsqueeze(0), counts.unsqueeze(-1)
        )
        sorted_cluster_value, sorted_cluster_indices = torch.sort(cluster)
        patch2point_map[patch2point_mask] = sorted_cluster_indices

        # generate cross masks
        assert self.mask_rate <= 0.5
        patch_mask = torch.zeros(patch_num, device=union_origin_coord.device).int()
        rand_perm = torch.randperm(patch_num)
        mask_patch_num = int(patch_num * self.mask_rate)

        # mask1 tag with 1, mask2 tag with 2
        patch_mask[rand_perm[0:mask_patch_num]] = 1
        patch_mask[rand_perm[mask_patch_num : mask_patch_num * 2]] = 2
        point_mask = torch.zeros(
            union_origin_coord.shape[0], device=union_origin_coord.device
        ).int()
        point_mask[
            patch2point_map[patch_mask == 1][patch2point_mask[patch_mask == 1]]
        ] = 1
        point_mask[
            patch2point_map[patch_mask == 2][patch2point_mask[patch_mask == 2]]
        ] = 2

        # separate mask to view1 and view2
        point_mask_split = point_mask.split(
            list(
                torch.cat(
                    [view1_batch_count.unsqueeze(-1), view2_batch_count.unsqueeze(-1)],
                    dim=-1,
                ).flatten()
            )
        )
        view1_point_mask = torch.cat(point_mask_split[0::2]) == 1
        view2_point_mask = torch.cat(point_mask_split[1::2]) == 2
        return view1_point_mask, view2_point_mask

    @torch.no_grad()
    def match_contrastive_pair(
        self, view1_coord, view1_offset, view2_coord, view2_offset, max_k, max_radius
    ):
        index, distance = pointops.knn_query(
            max_k,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )
        index = torch.cat(
            [
                torch.arange(index.shape[0], device=index.device, dtype=torch.long)
                .view(-1, 1, 1)
                .expand(-1, max_k, 1),
                index.view(-1, max_k, 1),
            ],
            dim=-1,
        )[distance.squeeze(-1) < max_radius]
        unique, count = index[:, 0].unique(return_counts=True)
        select = (
            torch.cumsum(count, dim=0)
            - torch.randint(count.max(), count.shape, device=count.device) % count
            - 1
        )
        index = index[select]
        if index.shape[0] > self.matching_max_pair:
            index = index[torch.randperm(index.shape[0])[: self.matching_max_pair]]
        return index

    def compute_contrastive_loss(
        self, view1_feat, view1_offset, view2_feat, view2_offset, match_index
    ):
        assert view1_offset.shape == view2_offset.shape
        
        view1_feat = view1_feat[match_index[:, 0]]
        view2_feat = view2_feat[match_index[:, 1]]
        view1_feat = view1_feat / (
            torch.norm(view1_feat, p=2, dim=1, keepdim=True) + 1e-7
        )
        view2_feat = view2_feat / (
            torch.norm(view2_feat, p=2, dim=1, keepdim=True) + 1e-7
        )
        sim = torch.mm(view1_feat, view2_feat.transpose(1, 0))

        with torch.no_grad():
            pos_sim = torch.diagonal(sim).mean()
            neg_sim = sim.mean(dim=-1).mean() - pos_sim / match_index.shape[0]
        labels = torch.arange(sim.shape[0], device=view1_feat.device).long()
        loss = self.nce_criteria(torch.div(sim, self.nce_t), labels)

        if get_world_size() > 1:
            dist.all_reduce(loss)
            dist.all_reduce(pos_sim)
            dist.all_reduce(neg_sim)
        return (
            loss / get_world_size(),
            pos_sim / get_world_size(),
            neg_sim / get_world_size(),
        )

    def forward(self, data_dict):
        total_loss = 0.0

        # shape:[10000, 3]
        # tensor([  [0.4466, 0.0245, 0.0412],
        #           [0.4399, 0.0207, 0.0390]])
        view1_origin_coord = data_dict["view1_origin_coord"]
        
        # shape:[10000, 3]
        # tensor([  [-0.0622, 0.0688, 0.0390],
        #           [-0.0774, 0.0666, 0.0361]])
        view1_coord = data_dict["view1_coord"]
        
        # shape:[10000, 6]
        # tensor([  [-0.4568, -0.7104, -0.8174, 0.5063, 0.8505, -0.1421],
        #           [-0.5058, -0.6798, -0.6015, -0.1276, 0.7727, 0.6218]])
        view1_feat = data_dict["view1_feat"]
        
        # shape:[2]
        # tensor([5000, 10000])
        view1_offset = data_dict["view1_offset"].int()

        view2_origin_coord = data_dict["view2_origin_coord"]
        view2_coord = data_dict["view2_coord"]
        view2_feat = data_dict["view2_feat"]
        view2_offset = data_dict["view2_offset"].int()
        
        # NOTE Masked Variables added NOTE #
        view1_masked_feat = data_dict.get("view1_masked_feat", data_dict["view1_feat"])
        view2_masked_feat = data_dict.get("view2_masked_feat", data_dict["view2_feat"])

        # NOTE Masked Functions added NOTE #
        view1_point_mask, view2_point_mask = self.generate_cross_masks(
            view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
        )

        view1_mask_tokens = self.mask_token.expand(view1_coord.shape[0], -1)
        view1_weight = view1_point_mask.unsqueeze(-1).type_as(view1_mask_tokens)
        view1_masked_feat = view1_masked_feat * (1 - view1_weight) + view1_mask_tokens * view1_weight

        view2_mask_tokens = self.mask_token.expand(view2_coord.shape[0], -1)
        view2_weight = view2_point_mask.unsqueeze(-1).type_as(view2_mask_tokens)
        view2_masked_feat = view2_masked_feat * (1 - view2_weight) + view2_mask_tokens * view2_weight

        # # union origin coord
        # shape:[10000]
        # tensor([0, 0, 0, 0, ..., 1, 1, 1, 1])
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        # shape:[2]
        # tensor([5000, 5000])
        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()
        
        # shape: ([5000, 3], [5000, 3])
        view1_xyz_split = view1_coord.split(list(view1_batch_count))
        view2_xyz_split = view2_coord.split(list(view2_batch_count))
        
        # shape: ([5000, 3], [5000, 3])
        transformed_points_X1_dict = {i: pts for i, pts in enumerate(view1_xyz_split)}
        transformed_points_X2_dict = {i: pts for i, pts in enumerate(view2_xyz_split)}
        
        # (dict1, dict2)
        # dict1: {
        # points [5000, 3]: [[249, 228, 183], [248, 229, 179], ...]
        # points_indices [5000, 1]: [0, 1, 2, 3, ..., 4999]
        # subregions []:
        # batch_idx [1]: 0
        # }
        batch_hierarchical_regions = data_dict['regions']


        view1_data_dict = dict(
            origin_coord=view1_origin_coord,
            coord=transformed_points_X1_dict,
            feat=view1_feat,
            masked_feat=view1_masked_feat, # NOTE Masked Variables added NOTE #
            offset=view1_offset,
        )
        view2_data_dict = dict(
            origin_coord=view2_origin_coord,
            coord=transformed_points_X2_dict,
            feat=view2_feat,
            masked_feat=view2_masked_feat, # NOTE Masked Variables added NOTE #
            offset=view2_offset,
        )
        
        # view1_data_dict = dict(
        #     origin_coord=view1_origin_coord,
        #     coord=view1_coord,
        #     feat=view1_feat,
        #     offset=view1_offset,
        # )
        # view2_data_dict = dict(
        #     origin_coord=view2_origin_coord,
        #     coord=view2_coord,
        #     feat=view2_feat,
        #     offset=view2_offset,
        # )

        # SparseConv based method need grid coord
        if "view1_grid_coord" in data_dict.keys():
            view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
        if "view2_grid_coord" in data_dict.keys():
            view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

        # view1_feat = self.point_encoder(view1_data_dict)
        # view2_feat = self.point_encoder(view2_data_dict)

        # # NOTE Masked Functions added NOTE # TODO: CAN WE MIX HERE OR SHOULD WE MIX IN THE FUNCTIONS OR DO WE NOT MIX AT ALL ?
        # # view mixing strategy
        # if random.random() < self.view1_mix_prob:
        #     view1_data_dict["offset"] = torch.cat(
        #         [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0
        #     )
        # if random.random() < self.view2_mix_prob:
        #     view2_data_dict["offset"] = torch.cat(
        #         [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
        #     )
        
        # # Encode and process with shared encoder using the same regions
        # # encode_and_propagate(batch_hierarchical_regions, self.point_encoder, self.aggregator, self.propagation_method, transformed_points_X1_dict,output_dim=self.output_dim)
        encode_and_propagate(batch_hierarchical_regions, self.point_encoder, self.aggregator, 
                             self.propagation_method, view_data_dict=view1_data_dict, output_dim=self.output_dim)
        encode_and_aggregate(batch_hierarchical_regions, self.point_encoder, self.aggregator, 
                             view_data_dict=view2_data_dict, max_levels=self.max_levels, output_dim=self.output_dim)

        # Compute loss for this sample
        #LOSS per level
        if self.loss_method in ["level"]:
            # Initialize dictionaries for accumulating features across batches
            all_features_dict_branch1 = {}
            all_features_dict_branch2 = {}
            for i in range(len(view1_offset)):
                hierarchical_regions = batch_hierarchical_regions[i] # Tree of (levelN, D)
                # Collect features
                features_dict_branch1 = {} # shape: [96]
                features_dict_branch2 = {} # shape: [96]
                collect_region_features_per_level(hierarchical_regions, features_dict_branch1, features_dict_branch2)

                # Combine features across batches
                combine_features(all_features_dict_branch1, features_dict_branch1)
                combine_features(all_features_dict_branch2, features_dict_branch2)
            loss = compute_contrastive_loss_per_level(all_features_dict_branch1, all_features_dict_branch2)
        #LOSS per point
        elif self.loss_method in ["point"]:
            # Initialize dictionaries for accumulating features across batches
            all_features_points_dict_branch1 = {}
            all_features_points_dict_branch2 = {}
            for i in range(len(view1_offset)):
                hierarchical_regions = batch_hierarchical_regions[i] # Tree of (levelN, D)
                # Collect features
                features_dict_points_branch1 = {} # shape: [5000, 96]
                features_dict_points_branch2 = {} # shape: [1, 96] NOTE should be [5000, 96]
                collect_region_features_per_points(hierarchical_regions,features_dict_points_branch1,features_dict_points_branch2)

                # Combine features across batches
                # all_features_points_dict_branch1 shape: [1, [4, 5000, 96]] [mx_lvl, [B, N, output_dim]]
                combine_features(all_features_points_dict_branch1, features_dict_points_branch1)
                combine_features(all_features_points_dict_branch2, features_dict_points_branch2)
            loss = compute_contrastive_loss_per_points(all_features_points_dict_branch1, all_features_points_dict_branch2)
        elif self.loss_method in ["masked"]:
            # view1_origin_coord = data_dict["view1_origin_coord"]
            # view1_coord = data_dict["view1_coord"]
            # view1_feat = data_dict["view1_feat"]
            # view1_offset = data_dict["view1_offset"].int()

            # view2_origin_coord = data_dict["view2_origin_coord"]
            # view2_coord = data_dict["view2_coord"]
            # view2_feat = data_dict["view2_feat"]
            # view2_offset = data_dict["view2_offset"].int()

            # # mask generation by union original coord (without spatial augmentation)
            # view1_point_mask, view2_point_mask = self.generate_cross_masks(
            #     view1_origin_coord, view1_offset, view2_origin_coord, view2_offset
            # )

            # view1_mask_tokens = self.mask_token.expand(view1_coord.shape[0], -1)
            # view1_weight = view1_point_mask.unsqueeze(-1).type_as(view1_mask_tokens)
            # view1_feat = view1_feat * (1 - view1_weight) + view1_mask_tokens * view1_weight

            # view2_mask_tokens = self.mask_token.expand(view2_coord.shape[0], -1)
            # view2_weight = view2_point_mask.unsqueeze(-1).type_as(view2_mask_tokens)
            # view2_feat = view2_feat * (1 - view2_weight) + view2_mask_tokens * view2_weight

            # view1_data_dict = dict(
            #     origin_coord=view1_origin_coord,
            #     coord=view1_coord,
            #     feat=view1_feat,
            #     offset=view1_offset,
            # )
            # view2_data_dict = dict(
            #     origin_coord=view2_origin_coord,
            #     coord=view2_coord,
            #     feat=view2_feat,
            #     offset=view2_offset,
            # )

            # # SparseConv based method need grid coord
            # if "view1_grid_coord" in data_dict.keys():
            #     view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
            # if "view2_grid_coord" in data_dict.keys():
            #     view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

            # # view mixing strategy
            # if random.random() < self.view1_mix_prob:
            #     view1_data_dict["offset"] = torch.cat(
            #         [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0
            #     )
            # if random.random() < self.view2_mix_prob:
            #     view2_data_dict["offset"] = torch.cat(
            #         [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            #     )

            # view1_feat = self.point_encoder(view1_data_dict)
            # view2_feat = self.point_encoder(view2_data_dict)
            
            match_index = self.match_contrastive_pair(
                view1_origin_coord,
                view1_offset,
                view2_origin_coord,
                view2_offset,
                max_k=self.matching_max_k,
                max_radius=self.matching_max_radius,
            )
            
            # Initialize dictionaries for accumulating features across batches
            all_masked_features_points_dict_branch1 = {}
            all_masked_features_points_dict_branch2 = {}
            for i in range(len(view1_offset)):
                hierarchical_regions = batch_hierarchical_regions[i] # Tree of (levelN, D)
                # Collect features
                masked_features_dict_points_branch1 = {} # shape: [5000, 96]
                masked_features_dict_points_branch2 = {} # shape: [1, 96] NOTE should be [5000, 96]
                collect_region_masked_features_per_points(hierarchical_regions, masked_features_dict_points_branch1, masked_features_dict_points_branch2)

                # Combine features across batches
                # all_masked_features_points_dict_branch1 shape: [1, [4, 5000, 96]] [mx_lvl, [B, N, output_dim]]
                combine_features(all_masked_features_points_dict_branch1, masked_features_dict_points_branch1)
                combine_features(all_masked_features_points_dict_branch2, masked_features_dict_points_branch2)
            
            all_masked_features_points_dict_branch1 = torch.cat(all_masked_features_points_dict_branch1[0], dim=0)
            all_masked_features_points_dict_branch2 = torch.cat(all_masked_features_points_dict_branch2[0], dim=0)
            nce_loss, pos_sim, neg_sim = self.compute_contrastive_loss(
                all_masked_features_points_dict_branch1, view1_offset, all_masked_features_points_dict_branch2, view2_offset, match_index
            )
            loss = nce_loss * self.contrast_weight
            result_dict = dict(nce_loss=nce_loss, pos_sim=pos_sim, neg_sim=neg_sim)

            # if self.color_head is not None:
            #     assert "view1_color" in data_dict.keys()
            #     assert "view2_color" in data_dict.keys()
            #     view1_color = data_dict["view1_color"]
            #     view2_color = data_dict["view2_color"]
            #     view1_color_pred = self.color_head(view1_feat[view1_point_mask])
            #     view2_color_pred = self.color_head(view2_feat[view2_point_mask])
            #     color_loss = (
            #         torch.sum((view1_color_pred - view1_color[view1_point_mask]) ** 2)
            #         + torch.sum((view2_color_pred - view2_color[view2_point_mask]) ** 2)
            #     ) / (view1_color_pred.shape[0] + view2_color_pred.shape[0])
            #     loss = loss + color_loss * self.reconstruct_weight
            #     result_dict["color_loss"] = color_loss

            # if self.normal_head is not None:
            #     assert "view1_normal" in data_dict.keys()
            #     assert "view2_normal" in data_dict.keys()
            #     view1_normal = data_dict["view1_normal"]
            #     view2_normal = data_dict["view2_normal"]
            #     view1_normal_pred = self.normal_head(view1_feat[view1_point_mask])
            #     view2_normal_pred = self.normal_head(view2_feat[view2_point_mask])

            #     view1_normal_pred = view1_normal_pred / (
            #         torch.norm(view1_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            #     )
            #     view2_normal_pred = view2_normal_pred / (
            #         torch.norm(view2_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            #     )
            #     normal_loss = (
            #         torch.sum(view1_normal_pred * view1_normal[view1_point_mask])
            #         + torch.sum(view2_normal_pred * view2_normal[view2_point_mask])
            #     ) / (view1_normal_pred.shape[0] + view2_normal_pred.shape[0])
            #     loss = loss + normal_loss * self.reconstruct_weight
            #     result_dict["normal_loss"] = normal_loss

            result_dict["loss"] = loss    
        
        total_loss += loss
        result_dict = dict(loss=total_loss)
        return result_dict