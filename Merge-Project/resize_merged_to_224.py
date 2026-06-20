import os
from PIL import Image
import shutil

# Source and destination directories
source_dir = 'merged'
dest_dir = 'resized_merged'

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    for dir_name in dirs:
        # Create corresponding directory in destination
        dest_subdir = os.path.join(dest_dir, os.path.relpath(os.path.join(root, dir_name), source_dir))
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)
    
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            source_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, os.path.relpath(source_path, source_dir))
            
            try:
                # Open image
                img = Image.open(source_path)
                # Resize to 224x224
                img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                # Save resized image
                img_resized.save(dest_path)
                print(f"Resized and saved: {dest_path}")
            except Exception as e:
                print(f"Error processing {source_path}: {e}")

print("Resizing completed.")