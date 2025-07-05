import os
import shutil

# Original and target directories
original_dir = "dataset"
reversed_dir = "dataset_reversed"

# Create the reversed directory if it doesn't exist
if not os.path.exists(reversed_dir):
    os.makedirs(reversed_dir)
    print(f"Created directory: {reversed_dir}")

# Define the order (7 to 1)
folders = [
    "tangrams_7_piece",
    "tangrams_6_piece",
    "tangrams_5_piece",
    "tangrams_4_piece",
    "tangrams_3_piece",
    "tangrams_2_piece",
    "tangrams_1_piece"
]

# Create symbolic links in the new directory
for folder in folders:
    source_path = os.path.join(original_dir, folder)
    target_path = os.path.join(reversed_dir, folder)
    
    # Create the folder and copy the files
    if os.path.exists(source_path):
        # Option 1: Symlink (saves disk space)
        # os.symlink(source_path, target_path)
        
        # Option 2: Copy the files (more compatible)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        for file in os.listdir(source_path):
            src_file = os.path.join(source_path, file)
            dst_file = os.path.join(target_path, file)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                
        print(f"Copied {folder} to {target_path}")
    else:
        print(f"Warning: Source folder {source_path} not found!")

print(f"\nReversed dataset structure created at {reversed_dir}")
print("Folder order:")
for folder in os.listdir(reversed_dir):
    if os.path.isdir(os.path.join(reversed_dir, folder)):
        file_count = len(os.listdir(os.path.join(reversed_dir, folder)))
        print(f"  - {folder} ({file_count} files)") 