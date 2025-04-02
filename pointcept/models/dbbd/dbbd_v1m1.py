"""
Pretraining TODO

Author: Anthony Yaghi, Manuel Philipp Vogel
"""

# External Libraries
import warnings
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

# Set manual seed for reproducibility
torch.manual_seed(12)

# Pointcept Modules
from pointcept.models.dbbd.Aggregator import MaxPoolAggregator
from pointcept.models.dbbd.Propagator import ConcatPropagation
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.utils.comm import get_world_size

def inference(encoder, points_tensor, view_data_dict=None, indices_list=None):
    # Encode the points using the dynamic encoder
    device = points_tensor.device
    resized_points_tensor = points_tensor.reshape(points_tensor.shape[0] *  points_tensor.shape[1], points_tensor.shape[2]) # (B, N, D) -> (B*N, D)
    
    offset_arr = []
    for i in range(points_tensor.shape[0]):
        offset_arr.append((i+1)*points_tensor.shape[1]) # Each offset is the number of points in the previous batch
    offset_arr = torch.tensor(offset_arr, device=device)
    
    indices = np.concatenate(indices_list, axis=0)
    grid_coord = view_data_dict["grid_coord"][indices]
    feat = view_data_dict["feat"][indices]

    points_dict = {"feat": feat, "coord": resized_points_tensor[:, :3], "grid_coord": grid_coord, 
                   "offset": offset_arr}
    
    # shape: [20000, 96] [B*N, output_dim]
    point_features = encoder(points_dict) # (B*N, output_dim)

    # shape: [20000] [B*N]
    batch = offset2batch(offset_arr)
    
    # shape: [4] [B]
    # tensor([5000, 5000, 5000, 5000]) [N, N, N, N]
    # shape: [B, N]
    batch_count = batch.bincount()
    point_features_split = point_features.split(list(batch_count))
    point_features_split = torch.stack(point_features_split)
    
    # shape: [4, 5000, 96] [B, N, output_dim]
    return point_features_split


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
    indices_list = []
    for i, reg in enumerate(region):
        batch_idx = reg['batch_idx']
        
        # shape: [5000, 3] [N, D]
        corresponding_transformed_points = view_data_dict["coord"][batch_idx]
        # shape: [5000] [N]
        indices = np.array(reg['points_indices'], dtype=int) # (levelN,)
  
        if len(indices) == 0:
            raise Exception("Got a region with no indics")
        else:
            indices_list.append(indices)
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
    
    # shape: [4, 5000, 96] [B, N, output_dim]
    batched_point_features = inference(encoder, batched_tensor, view_data_dict, indices_list=indices_list)

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
        reg['level_branch1'] = level 

        # Duplicate in parent_feature array based on number of upcoming subregions
        if len(reg['sub_regions']) <= 0:
            continue
            
        for sub_region in reg['sub_regions']:
            if len(sub_region['points_indices']) > 0:
                parent_feature_list.append(batched_region_feature[i])
                next_level_sub_regions.append(sub_region)
            else:
                warnings.warn("Sub region with no indices")
    if len(next_level_sub_regions) > 0 and len(parent_feature_list) > 0:
        assert len(next_level_sub_regions) == len(parent_feature_list), "Mismatch between next level subregions and parent features list"
        # print(f"PROPAGATION SUB REGIONS: {len(next_level_sub_regions)} at LEVEL: {level}")
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
        indices_list = []
        for reg in region:
            if reg['sub_regions']:
                for sub_region in reg['sub_regions']:
                    # Get the center of the sub-region
                    center_tensor = torch.tensor(sub_region['center'], dtype=torch.float32, device=view_data_dict['origin_coord'].device)
                    
                    # TODO: CHECK IF THIS IS CORRECT OR NEEDS ADJUSTMENT :TODO #
                    # Now search for matching index
                    index = torch.where(torch.all(view_data_dict['origin_coord'] == center_tensor, dim=1))[0]
                    if len(index) > 1:
                        index = np.array([index[0].cpu().numpy()])
                        # print(f"INDEX: {index}")
                        indices_list.append(index)
                    else:  
                        indices_list.append(index.cpu().numpy())
                    super_points_from_previous_level.append(sub_region['super_point_branch2'])
        

        # shape: [8, 96] [B * num_sample_lvl, output_dim]
        batched_tensor = torch.stack(super_points_from_previous_level) # (levelB, C)
        # shape: [8, 1, 96] [B * num_sample_lvl, 1, output_dim]
        batched_tensor = batched_tensor.unsqueeze(1) # (levelB, 1, C)

        # shape: [8, 1, 96] [B * num_sample_lvl, 1, output_dim]
        # print(f"AGGREGATION POINTS: {batched_tensor.shape} at LEVEL: {level}")
        batched_point_features = inference(encoder, batched_tensor, view_data_dict, indices_list=indices_list)

        # Aggregate
        # shape [8, 96] [B * num_sample_lvl, output_dim]
        batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)

        for i, reg in enumerate(region):
            reg['super_point_branch2'] = batched_region_feature[i] # (output_dim,)
            reg['level_branch2'] = level
            reg['super_point2'] = batched_point_features[i] 
            
    else:
        # IF LAST LEVEL
        points_tensor_list = []
        indices_list = []
        for i, reg in enumerate(region):
            batch_idx = reg['batch_idx']
            
            # shape: [5000, 3] [N, D]
            corresponding_transformed_points = view_data_dict["coord"][batch_idx]
            
            # shape: [5000] [N]
            indices = np.array(reg['points_indices'], dtype=int) # (levelN,)
    
            if len(indices) == 0:
                raise Exception("Got a region with no indics")
            else:
                indices_list.append(indices)
                # shape: [5000, 3] [N, D]
                points_tensor = corresponding_transformed_points[indices] # (levelN, D)
                # shape: [5000, 96] [N, output_dim]
                points_tensor = F.pad(points_tensor, (0, 99 - 2*points_tensor.shape[1]))
                
                # shape: [4 x [500, 96]] [B x [N, output_dim]]
                points_tensor_list.append(points_tensor)

        # shape: [4, 500, 96] [B, N, output_dim]
        batched_tensor = torch.stack(points_tensor_list) # (levelB, levelN, D or output_dim) # Assuming all regions on a level have the same number of points

        # shape: [4, 500, 96] [B, N, output_dim]
        # print(f"AGGREGATION POINTS: {batched_tensor.shape} at LEVEL: {level}")
        batched_point_features = inference(encoder, batched_tensor, view_data_dict, indices_list=indices_list)
        
        # shape: [4, 96] [B, output_dim]
        batched_region_feature = aggregator(batched_point_features) # (levelB, output_dim,)

        for i, reg in enumerate(region):
            reg['super_point_branch2'] = batched_region_feature[i] # (output_dim,)
            reg['super_point2'] = batched_point_features[i] # (N, output_dim)
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
        level1 = region['level_branch1']
        if level1 not in features_dict_branch1:
            features_dict_branch1[level1] = []
        # total desired shape: [1 x [5000, 96]] [mx_lvl x [N, output_dim]]
        features_dict_branch1[level1].append(region['super_point1']) # region['super_point1']: shape: [5000, 96] [N, output_dim]

    # Collect features from Branch 2
    if 'super_point2' in region:
        level2 = region['level_branch2']
        if level2 not in features_dict_branch2:
            features_dict_branch2[level2] = []
        # total desired shape: [1 x [5000, 96]] [mx_lvl x [N, output_dim]]
        features_dict_branch2[level2].append(region['super_point2']) # NOTE region['super_point2']: shape: [1, 96] BUT WE WANT [N, output_dim]

    # Recursively collect from sub-regions
    for sub_region in region['sub_regions']:
        collect_region_features_per_points(sub_region, features_dict_branch1, features_dict_branch2)

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

def compute_contrastive_loss_all_points(view1_point_feat: torch.Tensor,
                                        view2_point_feat: torch.Tensor,
                                        temperature: float = 0.07, device="cuda") -> torch.Tensor:
    if view1_point_feat.shape != view2_point_feat.shape:
        print(f"Mismatch of point features: {view1_point_feat.shape} vs {view2_point_feat.shape}")
        return None
    
    # Move to device
    view1_point_feat = view1_point_feat.to(device)
    view2_point_feat = view2_point_feat.to(device)

    # Normalize features along feature dimension (dim=1) because shape = [B*N, output_dim]
    features_branch1 = F.normalize(view1_point_feat, dim=1)
    features_branch2 = F.normalize(view2_point_feat, dim=1)

    # Compute similarity scores (cosine similarity)
    logits = torch.matmul(features_branch1, features_branch2.T) / temperature  # Shape: [B*N, B*N]

    # Generate labels: Positive pairs are along the diagonal
    num_points = logits.shape[0]  # B*N
    labels = torch.arange(num_points, device=device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Compute contrastive loss
    loss = criterion(logits, labels)

    return loss  # Returns a scalar tensor (final loss)

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
        alpha,
        beta
    ):
        super().__init__()
        self.point_encoder = build_model(backbone)
        self.aggregator = MaxPoolAggregator().to(device)
        self.propagation_method = ConcatPropagation().to(device)
        
        # Feature dimensionality update
        self.propagation_method.update_feature_dim(input_dim=99, feature_dim=96)
        
        self.output_dim = output_dim
        self.num_samples_per_level=num_samples_per_level
        self.max_levels=max_levels
        self.loss_method = loss_method
        
        # Weights for loss
        self.alpha = torch.tensor(alpha, device="cuda")
        self.beta = torch.tensor(beta, device="cuda")

    def compute_contrastive_loss(
        self, view1_feat, view1_offset, view2_feat, view2_offset, match_index
    ):
        assert view1_offset.shape == view2_offset.shape
        
        # Select matched features
        view1_feat = view1_feat[match_index[:, 0]]
        view2_feat = view2_feat[match_index[:, 1]]
        
        # Normalize matched features
        view1_feat = view1_feat / (
            torch.norm(view1_feat, p=2, dim=1, keepdim=True) + 1e-7
        )
        view2_feat = view2_feat / (
            torch.norm(view2_feat, p=2, dim=1, keepdim=True) + 1e-7
        )
        
        sim = torch.mm(view1_feat, view2_feat.transpose(1, 0))
        labels = torch.arange(sim.shape[0], device=view1_feat.device).long()
        
        # Compute positive/negative similarities
        with torch.no_grad():
            pos_sim = torch.diagonal(sim).mean()
            neg_sim = sim.mean(dim=-1).mean() - pos_sim / match_index.shape[0]
            
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
        view1_origin_coord = data_dict["view1_origin_coord"]
        
        # shape:[10000, 3]
        view1_coord = data_dict["view1_coord"]
        
        # shape:[10000, 6]
        view1_feat = data_dict["view1_feat"]
        
        # shape:[2]
        # tensor([5000, 10000])
        view1_offset = data_dict["view1_offset"].int()

        view2_origin_coord = data_dict["view2_origin_coord"]
        view2_coord = data_dict["view2_coord"]
        view2_feat = data_dict["view2_feat"]
        view2_offset = data_dict["view2_offset"].int()

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
            offset=view1_offset,
        )
        view2_data_dict = dict(
            origin_coord=view2_origin_coord,
            coord=transformed_points_X2_dict,
            feat=view2_feat,
            offset=view2_offset,
        )

        # SparseConv based method need grid coord
        if "view1_grid_coord" in data_dict.keys():
            view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
        if "view2_grid_coord" in data_dict.keys():
            view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

        view1_point_feat = self.point_encoder(view1_data_dict)
        view2_point_feat = self.point_encoder(view2_data_dict)
        
        # Encode and process with shared encoder using the same regions
        encode_and_propagate(batch_hierarchical_regions, self.point_encoder, self.aggregator, 
                             self.propagation_method, view_data_dict=view1_data_dict, output_dim=self.output_dim)
        encode_and_aggregate(batch_hierarchical_regions, self.point_encoder, self.aggregator, 
                             view_data_dict=view2_data_dict, max_levels=self.max_levels, output_dim=self.output_dim)

        # Compute loss for this sample
        # LOSS per level
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
        
        if self.loss_method in ["point_and_level"]:
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
            level_loss = compute_contrastive_loss_per_level(all_features_dict_branch1, all_features_dict_branch2)
            point_loss = compute_contrastive_loss_all_points(view1_point_feat, view2_point_feat)
            
            # Move losses to the same device as the model parameters, this is important for distributed training.
            device = next(self.parameters()).device  # Get model's device
            level_loss = level_loss.to(device)
            point_loss = point_loss.to(device)
            
            loss = self.alpha * level_loss + self.beta * point_loss
            
        # LOSS per point
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
            
        total_loss += loss
        result_dict = dict(loss=total_loss)
        return result_dict
