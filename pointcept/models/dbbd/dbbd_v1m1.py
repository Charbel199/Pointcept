"""
Pretraining TODO

Author: Anthony Yaghi, Manuel Philipp Vogel
"""
from typing import List, Dict, Any
import numpy as np
import torch
torch.manual_seed(12)
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.dbbd.Aggregator import MaxPoolAggregator
from pointcept.models.dbbd.Propagator import ConcatPropagation

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch


def inference(encoder, points_tensor, padding_size):
    # Encode the points using the dynamic encoder
    device = points_tensor.device
    resized_points_tensor = points_tensor.reshape(points_tensor.shape[0] *  points_tensor.shape[1], points_tensor.shape[2]) # (B, N, D) -> (B*N, D)
    
    offset_arr = []
    for i in range(points_tensor.shape[0]):
        offset_arr.append((i+1)*points_tensor.shape[1]) # Each offset is the number of points in the previous batch
    offset_arr = torch.tensor(offset_arr,device=device)


    points_dict = {"feat": F.pad(resized_points_tensor, padding_size), "coord": resized_points_tensor[:, :3], "grid_size": 0.01,
                   "offset": offset_arr}
    point_features = encoder(points_dict)["feat"]
    point_features = F.pad(point_features, (0, 128 - 64))  # since the output is 64, pad to 128

    batch = offset2batch(offset_arr)
    batch_count = batch.bincount()
    point_features_split = point_features.split(list(batch_count))
    point_features_split = torch.stack(point_features_split)

    return point_features_split


def encode_and_propagate(region: List[Dict[str, Any]], # (levelB, ...)
                         encoder, 
                         aggregator,
                         propagation_method, 
                         transformed_points: Dict[int, np.ndarray], # (B, N0, 3)
                         parent_feature: List[torch.Tensor] = None, # (levelB, ) if there's a parent feature list, 
                         level: int = 0,
                         output_dim=128) -> List[Dict[str, Any]]:
    
    
    # Iterate through regions and get corresponding indices then points from transformed points -> List of points vectors (levelB, levelN, D)
    points_tensor_list = []
    for i, reg in enumerate(region):
        batch_idx = reg['batch_idx']
        corresponding_transformed_points = transformed_points[batch_idx]
        indices = np.array(reg['points_indices'], dtype=int) # (levelN,)
  
        if len(indices) == 0:
            raise Exception("Got a region with no indics")
        else:
            points_tensor = corresponding_transformed_points[indices] # (levelN, D)

            # Propagate if there's a parent (Get the parent superpoints from the hierarchy for each region in the list)
            if parent_feature is not None:
                points_tensor = propagation_method.propagate(parent_feature[i], points_tensor) # (levelN, output_dim)
            
            points_tensor_list.append(points_tensor)

    batched_tensor = torch.stack(points_tensor_list) # (levelB, levelN, D or output_dim) # Assuming all regions on a level have the same number of points

    # Encode
    if batched_tensor.shape[2] == 6:
    # if batched_tensor.shape[2] == 3:
        padding_size = (0, output_dim)  # pad the last dimension (0 elems in front, 128 after)
    elif batched_tensor.shape[2] ==output_dim:
        padding_size = (0, 6)
        # padding_size = (0, 3)
    else:
        print(f"PROBLEM WITH TENSOR SIZE {batched_tensor.shape}")
    
    batched_point_features = inference(encoder, batched_tensor, padding_size)

    # Aggregate
    batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)


    # Iterate through list of regions and set as the corresponding superpoints for the level
    parent_feature_list = []
    next_level_sub_regions = []
    for i, reg in enumerate(region):
        reg['super_point_branch1'] = batched_region_feature[i] # (output_dim,)
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
        encode_and_propagate(next_level_sub_regions, encoder, aggregator, propagation_method, transformed_points, parent_feature=parent_feature_list, level=level+1)
    
    return region

def encode_and_aggregate(region: List[Dict[str, Any]], # (levelB, ...) 
                         encoder, 
                         aggregator,
                         transformed_points: Dict[int, np.ndarray], # (B, N0, 3)
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

        encode_and_aggregate(previous_level_sub_regions, encoder, aggregator, transformed_points, level=level+1, max_levels= max_levels)

        super_points_from_previous_level = []
        for reg in region:
            if reg['sub_regions']:
                for sub_region in reg['sub_regions']:
                    super_points_from_previous_level.append(sub_region['super_point_branch2'])
        

        batched_tensor = torch.stack(super_points_from_previous_level) # (levelB, C)
        batched_tensor = batched_tensor.unsqueeze(1) # (levelB, 1, C)

        # Encode
        # if batched_tensor.shape[2] == 3:
        if batched_tensor.shape[2] == 6:
            padding_size = (0, output_dim)  # pad the last dimension (0 elems in front, 128 after)
        elif batched_tensor.shape[2] ==output_dim:
            padding_size = (0, 6)
            # padding_size = (0, 3)
        else:
            print(f"PROBLEM WITH TENSOR SIZE {batched_tensor.shape}")
    
        batched_point_features = inference(encoder, batched_tensor, padding_size)

        # Aggregate
        batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)

        for i, reg in enumerate(region):
            reg['super_point_branch2'] = batched_region_feature[i] # (output_dim,)
            reg['level_branch2'] = level 
            
    else:
        # IF LAST LEVEL
        points_tensor_list = []
        for i, reg in enumerate(region):
            batch_idx = reg['batch_idx']
            corresponding_transformed_points = transformed_points[batch_idx]
            indices = np.array(reg['points_indices'], dtype=int) # (levelN,)
    
            if len(indices) == 0:
                raise Exception("Got a region with no indics")
            else:
                points_tensor = corresponding_transformed_points[indices] # (levelN, D)
                points_tensor_list.append(points_tensor)

        batched_tensor = torch.stack(points_tensor_list) # (levelB, levelN, D or output_dim) # Assuming all regions on a level have the same number of points


        # Encode
        if batched_tensor.shape[2] == 6:
        # if batched_tensor.shape[2] == 3:
            padding_size = (0, output_dim)  # pad the last dimension (0 elems in front, 128 after)
        elif batched_tensor.shape[2] ==output_dim:
            padding_size = (0, 6)
            # padding_size = (0, 3)
        else:
            print(f"PROBLEM WITH TENSOR SIZE {batched_tensor.shape}")

        batched_point_features = inference(encoder, batched_tensor, padding_size)
        # Aggregate
        batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)


        for i, reg in enumerate(region):
            reg['super_point_branch2'] = batched_region_feature[i] # (output_dim,)
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
        device

    ):
        super().__init__()
        self.point_encoder = build_model(backbone)
        self.aggregator = MaxPoolAggregator().to(device)
        self.propagation_method = ConcatPropagation().to(device)
        self.propagation_method.update_feature_dim(input_dim=backbone["in_channels"], feature_dim=128)
        self.output_dim = output_dim
        self.num_samples_per_level=num_samples_per_level
        self.max_levels=max_levels

    def forward(self, data_dict):
        total_loss = 0.0

        offsets = data_dict["offset"]

        view1_coord = data_dict["view1_coord"]
        view1_color = data_dict["view1_color"]
        view1_offset = data_dict["view1_offset"].int()
        view2_coord = data_dict["view2_coord"]
        view2_color = data_dict["view2_color"]
        view2_offset = data_dict["view2_offset"].int()
        
        xyzrgb1 = torch.cat((view1_coord, view1_color), dim=1)
        xyzrgb2 = torch.cat((view2_coord, view2_color), dim=1)

        # union origin coord
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)

        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()
        # view1_coord_split = view1_coord.split(list(view1_batch_count))
        # view2_coord_split = view2_coord.split(list(view2_batch_count))
        view1_xyzrgb_split = xyzrgb1.split(list(view1_batch_count))
        view2_xyzrgb_split = xyzrgb2.split(list(view2_batch_count))
        transformed_points_X1_dict = {i: pts for i, pts in enumerate(view1_xyzrgb_split)}
        transformed_points_X2_dict = {i: pts for i, pts in enumerate(view2_xyzrgb_split)}
        # transformed_points_X1_dict = {i: pts for i, pts in enumerate(view1_coord_split)}
        # transformed_points_X2_dict = {i: pts for i, pts in enumerate(view2_coord_split)}
        batch_hierarchical_regions = data_dict['regions']

        # Encode and process with shared encoder using the same regions
        encode_and_propagate(batch_hierarchical_regions, self.point_encoder, self.aggregator, self.propagation_method, transformed_points_X1_dict,output_dim=self.output_dim)
        encode_and_aggregate(batch_hierarchical_regions, self.point_encoder, self.aggregator, transformed_points_X2_dict, max_levels=self.max_levels,output_dim=self.output_dim)

        # Initialize dictionaries for accumulating features across batches
        all_features_dict_branch1 = {}
        all_features_dict_branch2 = {}
        for i in range(len(offsets)):
            hierarchical_regions = batch_hierarchical_regions[i] # Tree of (levelN, D)
            # Collect features
            features_dict_branch1 = {}
            features_dict_branch2 = {}

            collect_region_features_per_level(hierarchical_regions, features_dict_branch1, features_dict_branch2)

            # Combine features across batches
            combine_features(all_features_dict_branch1, features_dict_branch1)
            combine_features(all_features_dict_branch2, features_dict_branch2)

        # Compute loss for this sample
        loss = compute_contrastive_loss_per_level(all_features_dict_branch1, all_features_dict_branch2)
        total_loss += loss
        result_dict = dict(loss=total_loss)
        return result_dict