import os
import numpy as np
import time
# import open3d as o3d
import matplotlib.pyplot as plt

def process_npy_files(root_dir, visualize: bool=False):
    required_files = {"color.npy", "coord.npy", "instance.npy", "normal.npy", "segment20.npy", "segment200.npy"}
    time.sleep(1)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the directory contains all the required files
        if required_files.issubset(filenames):
            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                
                # Load the .npy file
                data = np.load(file_path)
                
                if visualize and file_name == "coord.npy":
                    visualize_pointcloud(data)
                
                # if data.shape[0] != 30000:
                if True:
                    print(f"I am at: {filenames} data.shape[0] {data.shape[0]}")

# def visualize_pointcloud(points):
#     """
#     Visualizes a 3D point cloud using Open3D.
    
#     :param points: (N, 3) NumPy array containing point cloud coordinates.
#     """
#     if points.shape[1] != 3:
#         raise ValueError(f"Expected (N, 3) shape for point cloud, but got {points.shape}")

#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
    
#     o3d.visualization.draw_geometries([point_cloud])

def visualize_pointcloud(points):
    """
    Visualizes a 3D point cloud using Matplotlib.
    
    :param points: (N, 3) NumPy array containing point cloud coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', marker=".")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Point Cloud Visualization")
    plt.show()
         
                    
if __name__ == "__main__":
    process_npy_files('./data/scannet', visualize=True)