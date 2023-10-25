import os
import random
import shutil

#  path to "images" and "annotations" folders
images_folder = "./images"
annotations_folder = "./annotations"

# Define the path to your "data" folder
data_folder = "data"

# Create the "data" folder
os.makedirs(data_folder, exist_ok=True)

# Create the "train" and "val" folders within the "data" folder
train_folder = os.path.join(data_folder, "train")
val_folder = os.path.join(data_folder, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Create subfolders "images" and "labels" within the "train" and "val" folders
train_images_folder = os.path.join(train_folder, "images")
train_labels_folder = os.path.join(train_folder, "labels")
val_images_folder = os.path.join(val_folder, "images")
val_labels_folder = os.path.join(val_folder, "labels")

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# List all image files in the "images" folder
image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]

# Calculate the number of images to put in the validation set (20% of the total)
num_val_images = int(0.2 * len(image_files))

# Randomly select images for the validation set
val_image_files = random.sample(image_files, num_val_images)

# Copy the selected images and corresponding label files to the validation set
for image_file in val_image_files:
    base_name = os.path.splitext(image_file)[0]
    annotation_file = base_name + ".txt"
    
    src_image_path = os.path.join(images_folder, image_file)
    dest_image_path = os.path.join(val_images_folder, image_file)
    src_annotation_path = os.path.join(annotations_folder, annotation_file)
    dest_annotation_path = os.path.join(val_labels_folder, annotation_file)
    
    shutil.copy(src_image_path, dest_image_path)
    shutil.copy(src_annotation_path, dest_annotation_path)

# Copy the remaining images and label files to the training set
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    annotation_file = base_name + ".txt"
    
    src_image_path = os.path.join(images_folder, image_file)
    dest_image_path = os.path.join(train_images_folder, image_file)
    src_annotation_path = os.path.join(annotations_folder, annotation_file)
    dest_annotation_path = os.path.join(train_labels_folder, annotation_file)
    
    shutil.copy(src_image_path, dest_image_path)
    shutil.copy(src_annotation_path, dest_annotation_path)

print("Data split into training and validation sets with images and annotations in separate folders.")
