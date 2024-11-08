import numpy as np
from typing import *


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

def hierarchical_region_proposal(points: np.ndarray, num_samples_per_level: int, max_levels: int, batch_idx: int, max_num_points: int = 30000) -> Dict[str, Any]:
    """
    Generate hierarchical regions using FPS.

    :param points: (N, D) array of points (coordinates + attributes).
    :param num_samples_per_level: Number of points to sample at each level.
    :param max_levels: Maximum depth of the hierarchy.
    :param batch_idx: Batch index for tracking.
    :return: Hierarchical regions as a dictionary.
    """
    def recursive_fps(points: np.ndarray, level: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if level >= max_levels or len(points) <= num_samples_per_level:
            return None, []

        if level == 0:
            min_num_points_per_pointcloud = 2000
        elif level == 1:
            min_num_points_per_pointcloud = 50

        points_pos = points[:, :3]
        sampled_centers = farthest_point_sampling(points_pos, num_samples_per_level)
        regions_pts_indices = assign_points_to_regions(points_pos, sampled_centers)

        hierarchical_regions = []
        for center, region_indices in zip(sampled_centers, regions_pts_indices):

            if len(region_indices)<min_num_points_per_pointcloud:
                continue
            region_indices = np.random.choice(region_indices, size=min_num_points_per_pointcloud, replace=False)
            region_points = points[region_indices]  # (N_region, D)
            _, sub_regions = recursive_fps(region_points, level + 1)
            hierarchical_regions.append({
                'center': center,
                'points': region_points,
                'points_indices': region_indices,
                'sub_regions': sub_regions,
                'batch_idx': batch_idx
            })

        return sampled_centers, hierarchical_regions

    _, hierarchical_regions = recursive_fps(points, 0)
    return {
        'points': points,
        'points_indices': np.arange(len(points)),
        'sub_regions': hierarchical_regions,
        'batch_idx': batch_idx
    }