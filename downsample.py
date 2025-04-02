import os
import argparse
import numpy as np
from typing import Set

def process_npy_files(root_dir: str, output_dir: str, max_points: int = 30000) -> None:
    """
    Process .npy files in each subdirectory of `root_dir`. 
    Keeps only those folders where all required .npy files contain at least `max_points` rows.
    Downsamples those files to exactly `max_points` points and saves them to `output_dir`.

    Args:
        root_dir (str): Path to the dataset root directory to process.
        output_dir (str): Path where valid and downsampled folders are saved.
        max_points (int): Number of points to retain after downsampling.
    """
    required_files: Set[str] = {
        "color.npy", "coord.npy", "instance.npy", "normal.npy", "segment20.npy", "segment200.npy"
    }

    for dirpath, _, filenames in os.walk(root_dir):
        print(f"\nChecking folder: {dirpath}")

        if required_files.issubset(set(filenames)):
            print(f"Valid directory: {dirpath}")
            delete_directory = False

            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                data = np.load(file_path)
                if data.shape[0] < max_points:
                    print(f"{file_name} has fewer than {max_points} points.")
                    delete_directory = True
                    break

            if delete_directory:
                continue

            # Create destination directory
            relative_path = os.path.relpath(dirpath, root_dir)
            dest_dir = os.path.join(output_dir, relative_path)
            os.makedirs(dest_dir, exist_ok=True)

            # Shared indices for consistent downsampling across all files
            shared_indices = None

            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                data = np.load(file_path)

                if shared_indices is None:
                    shared_indices = np.random.choice(data.shape[0], max_points, replace=False)

                sampled_data = data[shared_indices]
                np.save(os.path.join(dest_dir, file_name), sampled_data)
                print(f"{file_name}: saved {sampled_data.shape[0]} points -> {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and downsample .npy datasets.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the input dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save valid downsampled folders.")
    parser.add_argument("--max_points", type=int, default=30000, help="Max number of points to keep per file.")

    args = parser.parse_args()

    process_npy_files(args.root_dir, args.output_dir, args.max_points)

# Example usage:
# template: python downsample.py --root_dir /path/to/dataset --output_dir /path/to/output --max_points 30000
# example: python downsample.py --root_dir ./data/scannet --output_dir ./data/scannet_downsampled --max_points 30000
