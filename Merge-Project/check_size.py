from PIL import Image
import os

# Use the resized dataset if available; otherwise fall back to merged.
project_root = os.path.abspath(os.path.dirname(__file__))
paths = [
    os.path.join(project_root, 'resized_merged'),
    os.path.join(project_root, 'merged')
]

image_dir = next((p for p in paths if os.path.isdir(p)), None)
if image_dir is None:
    raise FileNotFoundError('Neither resized_merged nor merged directory exists in the project root.')

print(f'Checking images in: {image_dir}')

sizes = {}
count = 0
errors = 0
for root_dir, _, files in os.walk(image_dir):
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(root_dir, file_name)
            try:
                with Image.open(file_path) as img:
                    sizes[img.size] = sizes.get(img.size, 0) + 1
                    count += 1
            except Exception as exc:
                print(f'Error opening {file_path}: {exc}')
                errors += 1

print(f'Total images checked: {count}')
for size, num in sorted(sizes.items()):
    print(f'  {size}: {num}')
if errors:
    print(f'Errors opening images: {errors}')
