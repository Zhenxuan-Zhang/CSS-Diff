import os
import glob
from PIL import Image
import numpy as np

def check_data_range(folder_path):
    # Define file patterns for real_B and fake_B images
    patterns = {
        'real_B': os.path.join(folder_path, "*real_B.png"),
        'fake_B': os.path.join(folder_path, "*fake_B.png")
    }
    
    # Process each set of images
    for key, pattern in patterns.items():
        file_list = glob.glob(pattern)
        if not file_list:
            print(f"No files found for pattern: {pattern}")
            continue
        
        print(f"\nProcessing {key} images:")
        for file_path in file_list:
            # Load image
            image = Image.open(file_path)
            # Convert the image to a NumPy array
            arr = np.array(image)
            # Calculate min and max values from the array
            min_val = arr.min()
            max_val = arr.max()
            # Report the result
            print(f"File: {os.path.basename(file_path)} -- min: {min_val}, max: {max_val}")

if __name__ == "__main__":
    # Specify the folder containing the images
    folder_path = "/media/NAS07/USER_PATH/peiyuan/PGD/pss_test_result/pss_unest_fth025_depth12_nld3_ssi0/test_best/low_high"
    check_data_range(folder_path)
