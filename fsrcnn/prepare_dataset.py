import os
from PIL import Image

base_dir = './data'
hr_dir = os.path.join(base_dir, 'HR')
lr_dir = os.path.join(base_dir, 'LR')

os.makedirs(hr_dir, exist_ok=True)
os.makedirs(lr_dir, exist_ok=True)

for file in os.listdir(base_dir):
    if file.endswith(('png', 'jpg', 'jpeg')):
        os.rename(os.path.join(base_dir, file), os.path.join(hr_dir, file))

hr_files = len(os.listdir(hr_dir))
print(f"Number of HR images: {hr_files}")

def downsample_image(img, scale=2):
    width, height = img.size
    new_size = (width // scale, height // scale)
    return img.resize(new_size, Image.BICUBIC)


for img_name in os.listdir(hr_dir):
    if img_name.endswith(('png', 'jpg', 'jpeg')):
        hr_img_path = os.path.join(hr_dir, img_name)
        hr_img = Image.open(hr_img_path)
        lr_img = downsample_image(hr_img)
        lr_img.save(os.path.join(lr_dir, img_name))

lr_images = len(os.listdir(lr_dir))
print(f"Number of LR images generated: {lr_images}")
