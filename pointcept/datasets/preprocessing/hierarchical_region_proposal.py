import numpy as np
from typing import *
import warnings


def assign_points_to_regions(points: np.ndarray, centers: np.ndarray) -> List[List[int]]:
    """
    Assign each point to the nearest region center.

    :param points: (N, 3) array of points.
    :param centers: (M, 3) array of region centers.
    :return: List of lists, each containing the indices of points in the corresponding region.
    """
    if points.size == 0 or centers.size == 0:
        return [[] for _ in range(len(centers))]

    num_points = points.shape[0]
    regions = [[] for _ in range(len(centers))]

    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)  # Shape: (N, M)
    nearest_centers = np.argmin(distances, axis=1)  # Shape: (N,)

    for i in range(num_points):
        regions[nearest_centers[i]].append(i)

    return regions


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Perform Farthest Point Sampling (FPS) on a set of points.

    :param points: (N, 3) array of point positions.
    :param num_samples: Number of points to sample.
    :return: (num_samples, 3) array of sampled point positions.
    """
    N, _ = points.shape
    sampled_indices = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)

    # Randomly select the first point
    selected_idx = np.random.randint(N)
    sampled_indices[0] = selected_idx

    for i in range(1, num_samples):
        current_point = points[selected_idx, :]
        dist = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist)
        selected_idx = np.argmax(distances)
        sampled_indices[i] = selected_idx

    sampled_points = points[sampled_indices]
    return sampled_points

def balance_regions(points: np.ndarray, regions: List[List[int]], centers: np.ndarray) -> List[List[int]]:
    """
    Redistribute points from overloaded regions to underpopulated ones.
    """
    num_regions = len(regions)
    total_points = sum(len(r) for r in regions)
    target_size = total_points // num_regions  # Ideal number of points per region

    # Sort points within each region by distance to their center
    sorted_regions = []
    for i, region in enumerate(regions):
        region_points = np.array(region)
        distances = np.linalg.norm(points[region_points] - centers[i], axis=1)
        sorted_indices = np.argsort(distances)  # Sort by increasing distance
        sorted_regions.append(region_points[sorted_indices].tolist())

    # Identify **underpopulated regions**
    underloaded_regions = {i for i in range(num_regions) if len(sorted_regions[i]) < target_size}
    
    # Move excess points from overloaded regions to underpopulated ones
    for i in range(num_regions):
        while len(sorted_regions[i]) > target_size and len(underloaded_regions) > 0:
            # Remove the farthest point
            farthest_point = sorted_regions[i].pop()
            
            # Find the closest center **only among underloaded regions**
            available_centers = list(underloaded_regions)
            
            # Find the closest center that still has space
            distances_to_centers = np.linalg.norm(points[farthest_point] - centers[available_centers], axis=1)
            new_center_idx = available_centers[np.argmin(distances_to_centers)]
            
            # Assign the point to the selected underloaded region
            sorted_regions[new_center_idx].append(farthest_point)
            
            # If the new region reaches the target size, remove it from the underloaded set
            if len(sorted_regions[new_center_idx]) >= target_size:
                underloaded_regions.remove(new_center_idx)

    return sorted_regions


def hierarchical_region_proposal(points: np.ndarray,points_rgb: np.ndarray, num_samples_per_level: int, max_levels: int, batch_idx: int,min_num_points_list:List[int], equal_splits:bool) -> Dict[str, Any]:
    """
    Generate hierarchical regions using FPS.

    :param points: (N, D) array of points (coordinates + attributes).
    :param points_rgb: (N, D) RGB.
    :param num_samples_per_level: Number of points to sample at each level.
    :param max_levels: Maximum depth of the hierarchy.
    :param batch_idx: Batch index for tracking.
    :return: Hierarchical regions as a dictionary.
    """
    def recursive_fps(points: np.ndarray, colors: np.ndarray, level: int, min_num_points_list:List[int], equal_splits:bool) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if level >= max_levels or len(points) <= num_samples_per_level:
            return None, []

        if level > len(min_num_points_list):
            raise Exception("The 'min_num_points_list' must contain values for all levels.")
        else:
            min_num_points_per_pointcloud = min_num_points_list[level]

        points_pos = points[:, :3]
        
        if not equal_splits:
            sampled_centers = farthest_point_sampling(points_pos, num_samples_per_level)
            regions_pts_indices = assign_points_to_regions(points_pos, sampled_centers)
        else:
            # Step 1: FPS selects well-separated centers
            sampled_centers = farthest_point_sampling(points_pos, num_samples_per_level)

            # Step 2: Assign points to nearest centers
            initial_regions = assign_points_to_regions(points_pos, sampled_centers)

            # Step 3: Balance the point distribution across regions
            regions_pts_indices = balance_regions(points_pos, initial_regions, sampled_centers)
        
        hierarchical_regions = []
        for center, region_indices in zip(sampled_centers, regions_pts_indices):
            if len(region_indices) < min_num_points_per_pointcloud and not equal_splits:
                # warnings.warn("Length of region_indices {} < {} min_num_points_per_pointcloud".format(len(region_indices), min_num_points_per_pointcloud))
                # extra_indices = np.random.choice(region_indices, size=(min_num_points_per_pointcloud - len(region_indices)), replace=True)
                # region_indices = np.concatenate([region_indices, extra_indices])
                # region_points = points[region_indices]  # (N_region, D)
                # region_colors = colors[region_indices]
                continue
            else:
                region_indices = np.random.choice(region_indices, size=min_num_points_per_pointcloud, replace=False)
                region_points = points[region_indices]  # (N_region, D)
                region_colors = colors[region_indices]
            _, sub_regions = recursive_fps(region_points, points_rgb, level + 1, min_num_points_list=min_num_points_list, equal_splits=equal_splits)
            hierarchical_regions.append({
                'center': center,
                'points': region_points,
                'colors': region_colors,
                'points_indices': region_indices,
                'sub_regions': sub_regions,
                'batch_idx': batch_idx
            })

        return sampled_centers, hierarchical_regions

    _, hierarchical_regions = recursive_fps(points, points_rgb, 0, min_num_points_list=min_num_points_list, equal_splits=equal_splits)
    return {
        'points': points,
        'points': points_rgb,
        'points_indices': np.arange(len(points)),
        'sub_regions': hierarchical_regions,
        'batch_idx': batch_idx
    }