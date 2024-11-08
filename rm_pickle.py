import os
import glob

def remove_pickle_files(directory):
    # Find all .pickle files in the directory and subdirectories
    pickle_files = glob.glob(os.path.join(directory, '**', '*.pickle'), recursive=True)
    
    # Remove each .pickle file found
    for file_path in pickle_files:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

# Usage
remove_pickle_files('data/s3dis')
