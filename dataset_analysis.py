import os
import numpy as np
import time

def process_npy_files(root_dir):
    required_files = {"color.npy", "coord.npy", "instance.npy", "normal.npy", "segment20.npy", "segment200.npy"}
    time.sleep(1)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the directory contains all the required files
        if required_files.issubset(filenames):
            for file_name in required_files:
                file_path = os.path.join(dirpath, file_name)
                
                # Load the .npy file
                data = np.load(file_path)
                
                # if data.shape[0] != 30000:
                if True:
                    print(f"I am at: {filenames} data.shape[0] {data.shape[0]}")
                    
                    
if __name__ == "__main__":
    process_npy_files('./data/scannet')