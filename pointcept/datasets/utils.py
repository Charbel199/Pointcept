"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

import open3d as o3d


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    region_batch = []
    for i, dictionary in enumerate(batch):
        region_dict = dictionary.pop("regions") if "regions" in dictionary else None
        if region_dict is not None:
            region_dict["batch_idx"] = i
        region_batch.append(region_dict)
    
    
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    batch["regions"] = region_batch
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))


def collect_regions_by_level(region, level=0, level_dict=None):
    """
    Recursively collect all regions grouped by hierarchy levels.

    :param region: Root region dictionary.
    :param level: Current level in the hierarchy.
    :param level_dict: Dictionary to store regions per level.
    """
    if level_dict is None:
        level_dict = {}

    if level not in level_dict:
        level_dict[level] = []

    level_dict[level].append(region['points_indices'])  # Store indices at this level

    # Recursively process subregions
    for sub_region in region.get('sub_regions', []):
        collect_regions_by_level(sub_region, level + 1, level_dict)

    return level_dict

def save_colored_regions(points, regions_by_level, filename="colored_regions.ply"):
    """
    Save processed point cloud with unique colors per hierarchy level and region.

    :param points: (N, 3) NumPy array of all points.
    :param regions_by_level: Dictionary containing point indices grouped by hierarchy levels.
    :param filename: Output filename.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    num_levels = len(regions_by_level)
    point_colors = np.zeros((points.shape[0], 3))  # Default color: black

    # Assign different colors per region within each level
    for level, regions in regions_by_level.items():
        num_regions = len(regions)
        region_colors = np.random.rand(num_regions, 3)  # Unique color for each region

        for i, region_indices in enumerate(regions):
            point_colors[region_indices] = region_colors[i]  # Assign region-specific color

    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

    os.makedirs("colored_regions", exist_ok=True)  # Ensure directory exists
    output_path = os.path.join("colored_regions", filename)
    o3d.io.write_point_cloud(output_path, point_cloud)

    print(f"Colored point cloud saved as {output_path}.")
