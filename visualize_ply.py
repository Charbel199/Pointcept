import os
import open3d as o3d

folder_name = "colored_regions_2_2_15k_7k_3k"

for colored_regions_file in os.listdir(folder_name):
    ply_file = os.path.join(folder_name, colored_regions_file)
    pcd = o3d.io.read_point_cloud(ply_file)
    print(f"Number of Points: {len(pcd.points)}")
    o3d.visualization.draw_geometries([pcd])
