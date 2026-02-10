import os
import re

def rename_images(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip if the directory doesn't end with 'A' or 'B'
        if not (dirpath.endswith("A") or dirpath.endswith("B")):
            continue

        # Decide the prefix
        if dirpath.endswith("A"):
            prefix = "low_field"
        elif dirpath.endswith("B"):
            prefix = "high_field"
        else:
            continue

        print(f"Renaming files in: {dirpath} -> prefix: {prefix}")

        for fname in sorted(filenames):
            if not fname.endswith(".png"):
                continue

            # Extract numerical part from filename (assumes filename is like 0000.png)
            base, ext = os.path.splitext(fname)
            try:
                number = int(base)
            except ValueError:
                print(f"Skipping {fname}, not a valid integer name.")
                continue

            new_name = f"{prefix}_{number}{ext}"
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dirpath, new_name)

            print(f"Renaming: {fname} -> {new_name}")
            os.rename(src_path, dst_path)

if __name__ == "__main__":
    root_directory = "/media/NAS07/USER_PATH/peiyuan/paired/images/T1"  # Change this to your root path
    rename_images(root_directory)
