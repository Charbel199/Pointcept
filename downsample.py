import os
import numpy as np
import shutil

def process_npy_files(root_dir):
    required_files = {"color.npy", "coord.npy", "instance.npy", "normal.npy", "segment20.npy", "segment200.npy"}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Checking folder: {dirpath}")

        # Check if the directory contains all the required files
        if required_files.issubset(filenames):
            print(f"Processing folder: {dirpath}")
            delete_directory = False  # Flag to track if directory should be deleted

            # First Pass: Check if any file has fewer than 30,000 points
            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                data = np.load(file_path)

                # If any file has fewer than 30,000 points, mark directory for deletion
                if data.shape[0] < 30000:
                    delete_directory = True
                    print(f"{file_name} in {dirpath} has fewer than 30,000 points. Marking directory for deletion.")
                    break  # No need to check further, delete the folder

            # Delete the directory if flagged
            if delete_directory:
                shutil.rmtree(dirpath)
                print(f"Deleted directory: {dirpath}")
                continue  # Move to the next folder

            # Second Pass: Downsample each file to exactly 30,000 points
            print(f"Downsampling all files in {dirpath} to 30,000 points.")

            shared_indices = None  # This ensures all files use the same sampled indices

            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                data = np.load(file_path)

                # Generate shared indices for sampling
                if shared_indices is None:
                    shared_indices = np.random.choice(data.shape[0], 30000, replace=False)

                # Downsample using shared indices
                sampled_data = data[shared_indices]

                # Overwrite the existing .npy file with the sampled data
                np.save(file_path, sampled_data)
                print(f"Overwritten {file_name} with {sampled_data.shape[0]} points.")

# Run the function
process_npy_files('./data/scannet')
