import os
import json
import shutil
import numpy as np
from PIL import Image
import tqdm 

from utils.utils import create_empty_dir

KNOWN_NAMES = ["person", "bench", "backpack", "handbag", "chair", "potted plant", "laptop", "mouse", "keyboard", "cell phone"]

CONE_NAMES = ["cone", "traffic cone"]

def copy_directory(src_dir, dest_dir):
    """
    Copy all files from src_dir to dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(src_dir):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copyfile(src, dst)

def copy_new_labels(new_image_dir, new_label_dir, train_image_dir, train_label_dir, val_image_dir, val_label_dir):

    # Get list of images
    image_names = os.listdir(new_image_dir)
    for image_name in tqdm.tqdm(image_names):
        # Get the image path
        image_path = os.path.join(new_image_dir, image_name)
        label_path = os.path.join(new_label_dir, image_name.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist.")
            continue

        # Select if train or val, 90% train, 10% val
        if np.random.rand() < 0.9:
            # Copy to train
            dest_image_dir = train_image_dir
            dest_label_dir = train_label_dir
        else:
            # Copy to val
            dest_image_dir = val_image_dir
            dest_label_dir = val_label_dir

        # Copy the image
        shutil.copy(image_path, dest_image_dir)

        # Read the label file
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Create a new label file
        new_label_path = os.path.join(dest_label_dir, image_name.replace(".jpg", ".txt"))
        with open(new_label_path, "w") as f:
            for line in lines:
                entries = line.split()
                
                # If number of entries is > 5, combine the first entries into one
                if len(entries) > 5:
                    class_name = " ".join(entries[:-4])
                    x_center = entries[-4]
                    y_center = entries[-3]
                    bbox_width = entries[-2]
                    bbox_height = entries[-1]
                else:
                    class_name = entries[0]
                    x_center = entries[1]
                    y_center = entries[2]
                    bbox_width = entries[3]
                    bbox_height = entries[4]

                if class_name in KNOWN_NAMES:
                    # Get the class id
                    class_id = KNOWN_NAMES.index(class_name)
                elif class_name in CONE_NAMES:
                    class_id = len(KNOWN_NAMES) + 1
                else:
                    class_id = 0
                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

if __name__ == "__main__":

    # Set the seed
    np.random.seed(88)

    # New Data directories 
    new_image_dir = "updated_data/cone_data/images"
    new_label_dir = "updated_data/cone_data/labels"

    # Directory to where we want to save the filtered images/labels
    train_image_dir = "updated_data/train/images"
    train_labels_dir = "updated_data/train/labels"

    val_image_dir = "updated_data/val/images"
    val_labels_dir = "updated_data/val/labels"

    test_image_dir = "updated_data/test/images"
    test_labels_dir = "updated_data/test/labels"

    original_train_image_dir = "data/train/images"
    original_train_labels_dir = "data/train/labels"
    original_val_image_dir = "data/val/images"
    original_val_labels_dir = "data/val/labels"
    original_test_image_dir = "data/test/images"
    original_test_labels_dir = "data/test/labels"

    # Create directories if they don't exist
    print("Creating/Emptying Directories...")
    create_empty_dir(train_image_dir)
    create_empty_dir(train_labels_dir)
    create_empty_dir(val_image_dir)
    create_empty_dir(val_labels_dir)
    create_empty_dir(test_image_dir)
    create_empty_dir(test_labels_dir)

    # Copy the images and labels from data/* to updated_data/*
    print("Copying images and labels...")
    print("Train...")
    copy_directory(original_train_image_dir, train_image_dir)
    copy_directory(original_train_labels_dir, train_labels_dir)
    print("Val...")
    copy_directory(original_val_image_dir, val_image_dir)
    copy_directory(original_val_labels_dir, val_labels_dir)
    print("Test...")
    copy_directory(original_test_image_dir, test_image_dir)
    copy_directory(original_test_labels_dir, test_labels_dir)

    # Now copy the new images and labels to the updated_data directory
    print("Copying new images and labels...")
    copy_new_labels(new_image_dir, new_label_dir, train_image_dir, train_labels_dir, val_image_dir, val_labels_dir)