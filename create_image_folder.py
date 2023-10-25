import os
import shutil


train_dir = "./train"

# Create a directory to store images
image_dir = os.path.join(train_dir, "images")
os.makedirs(image_dir, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]  

for file_name in os.listdir(train_dir):
    if file_name.lower().endswith(tuple(image_extensions)):
        image_path = os.path.join(train_dir, file_name)
        shutil.copy(image_path, os.path.join(image_dir, file_name))

print("Image files copied to the 'images' folder.")
