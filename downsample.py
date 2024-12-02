import os
import numpy as np
import shutil

def process_npy_files(root_dir):
    required_files = {"color.npy", "coord.npy", "instance.npy", "normal.npy", "segment.npy"}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"I am at: {filenames}")
        
        # Check if the directory contains all the required files
        if required_files.issubset(filenames):
            print(f"Processing folder: {dirpath}")
            delete_directory = False  # Flag to track if directory should be deleted

            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                
                # Load the .npy file
                data = np.load(file_path)

                # Check if there are fewer than 30,000 points
                if data.shape[0] < 30000:
                    delete_directory = True  # Set flag to delete the directory
                    print(f"{file_name} in {dirpath} has fewer than 30,000 points. Marking directory for deletion.")
                    break  # No need to check other files if one is below threshold

            # Delete the directory if flagged
            if delete_directory:
                shutil.rmtree(dirpath)
                print(f"Deleted directory: {dirpath}")
            else:
                # If directory is not flagged for deletion, sample data for each file
                for file_name in required_files:
                    file_path = os.path.join(dirpath, file_name)
                    data = np.load(file_path)

                    # Randomly select 30,000 points (or keep all points if there are fewer than 30,000)
                    if data.shape[0] >= 30000:
                        indices = np.random.choice(data.shape[0], 30000, replace=False)
                        sampled_data = data[indices]
                    else:
                        sampled_data = data

                    # Overwrite the existing .npy file with the sampled data
                    np.save(file_path, sampled_data)
                    print(f"Overwritten {file_name} with {sampled_data.shape[0]} points.")

# Usage
process_npy_files('./data/s3dis')
