import os
import json
import shutil
import numpy as np
from PIL import Image
import tqdm 

from utils.utils import create_empty_dir

# 1: person, 15: bench, 27: backpack, 31: handbag, 62: chair, 64: potted plant, 73: laptop, 74: mouse, 76: keyboard, 77: cell phone
COCO_LABELS = [1, 15, 27, 31, 62, 64, 73, 74, 76, 77]


""" 
This file is used to filter the images/filters that are used in this project.
Since we train on only a subset of COCO, we only need the images/labels associated
with the classes we are interested in.
"""

def read_coco_labels(labels_file):
    # Read the json file
    with open(labels_file) as f:
        data = json.load(f)
    

    labels = data["annotations"]
    num_labels = len(labels)

    images_with_labels = set()  # Keep track of images that have labels
    for i in range(num_labels):
        label = labels[i]
        image_id = label["image_id"]
        category = label["category_id"]

        # Keep track of images that have labels of interest
        if category in COCO_LABELS:
            images_with_labels.add(image_id)

    return images_with_labels


def copy_images_and_labels(src_image_dir, dest_image_dir, labels_file, dest_labels_dir, images_with_labels, split="train"):
    
    if split == "train":
        # We should choose 90% of the images for training, 10% for validation
        num_images = len(images_with_labels)
        num_train = int(0.9 * num_images)
        train_images = np.random.choice(list(images_with_labels), num_train, replace=False)
        val_images = images_with_labels - set(train_images)

        print("Copying \"train\" images and labels...")
        coco_train_dir = os.path.join(src_image_dir, "train2017")
        copy_images(train_images, coco_train_dir, os.path.join(dest_image_dir, "train", "images"))
        copy_labels(train_images, labels_file, coco_train_dir, os.path.join(dest_labels_dir, "train", "coco_labels"))

        print("Copying \"val\" images and labels...")
        copy_images(val_images, coco_train_dir, os.path.join(dest_image_dir, "val", "images"))
        copy_labels(val_images, labels_file, coco_train_dir, os.path.join(dest_labels_dir, "val", "labels"))  
    elif split == "val":  # Images from coco val are used for testing
        print("Copying \"test\" images and labels...")
        coco_val_dir = os.path.join(src_image_dir, "val2017")
        copy_images(images_with_labels, coco_val_dir, os.path.join(dest_image_dir, "test", "images"))
        copy_labels(images_with_labels, labels_file, coco_val_dir, os.path.join(dest_labels_dir, "test", "labels"))
    else: 
        # Not allowed, raise an error
        raise ValueError("Invalid split type. Choose either 'train' or 'val'.")


def copy_images(images, source_dir, dest_dir):
    print("Copying Images...")
    for image in tqdm.tqdm(images):
        image_name = str(image).zfill(12) + ".jpg"
        shutil.copy(os.path.join(source_dir, image_name), dest_dir)


def copy_labels(images, labels_file, image_dir, dest_dir):
    # Read the json file
    with open(labels_file) as f:
        data = json.load(f)

    labels = data["annotations"]
    num_labels = len(labels)

    print("Copying Labels...")
    for i in tqdm.tqdm(range(num_labels)):
        label = labels[i]
        image_id = label["image_id"]
        category = label["category_id"]

        # Keep track of images that have labels of interest
        if image_id in images:
            # Write or append to txt file in the format: class x_center y_center width height
            bbox = label["bbox"]
            x1, y1, width, height = bbox
            x2, y2 = x1 + width, y1 + height
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            # Get image size
            image_name = str(image_id).zfill(12) + ".jpg"
            image = Image.open(os.path.join(image_dir, image_name))
            image_width, image_height = image.size

            # Normalize the coordinates
            xc /= image_width
            yc /= image_height
            width /= image_width
            height /= image_height

            # Write to file
            label_file = os.path.join(dest_dir, image_name.replace(".jpg", ".txt"))
            if category in COCO_LABELS:
                remapped_category = COCO_LABELS.index(category) + 1  # +1 since reserve 0 for unknown objects
            else:
                remapped_category = 0
            with open(label_file, "a") as f:
                f.write(f"{remapped_category} {xc} {yc} {width} {height}\n")


def save_remapped_categories(annotations_file):

    # Open the annotations file
    with open(annotations_file) as f:
        data = json.load(f)

    categories = data["categories"]

    # Saves the remapped categories to a txt file
    with open("data/categories.txt", "w") as f:
        f.write("0 unknown\n")
        for i, category in enumerate(COCO_LABELS):
            for j in range(len(categories)):
                if categories[j]["id"] == category:
                    f.write(f"{i+1} {categories[j]['name']}\n")
                    break


def save_data_yaml_file(annotations_file):
    """
    Saves the yaml file needed for YOLO training
    """
    # Read the annotations file
    with open(annotations_file) as f:
        data = json.load(f)
    categories = data["categories"]


    with open("tools/data.yaml", "w") as f:
        f.write("# Data file for YOLO training\n")
        f.write("path: ../data\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images  # (optional)\n")

        f.write("\n")
        f.write("names:\n")
        f.write("  0: unknown\n")

        for i, category in enumerate(COCO_LABELS):
            for j in range(len(categories)):
                if categories[j]["id"] == category:
                    f.write(f"  {i+1}: {categories[j]['name']}\n")
                    break
        

if __name__ == "__main__":

    np.random.seed(42)

    # COCO 2017 dataset directories 
    coco_image_dir = "coco/2017/images"
    coco_annotation_dir = "coco/2017/annotations"

    # Directory to where we want to save the filtered images/labels
    train_image_dir = "data/train/images"
    train_labels_dir = "data/train/labels"
    train_coco_labels = "data/train/coco_labels"

    val_image_dir = "data/val/images"
    val_labels_dir = "data/val/labels"

    test_image_dir = "data/test/images"
    test_labels_dir = "data/test/labels"

    # Create directories if they don't exist
    print("Creating/Emptying Directories...")
    create_empty_dir(train_image_dir)
    create_empty_dir(train_labels_dir)
    create_empty_dir(train_coco_labels)
    create_empty_dir(val_image_dir)
    create_empty_dir(val_labels_dir)
    create_empty_dir(test_image_dir)
    create_empty_dir(test_labels_dir)

    # Read the COCO labels
    train_images = read_coco_labels(os.path.join(coco_annotation_dir, "instances_train2017.json"))
    val_images = read_coco_labels(os.path.join(coco_annotation_dir, "instances_val2017.json"))

    # Copy images and labels to the new directories (for YOLO)
    copy_images_and_labels(coco_image_dir, "data", os.path.join(coco_annotation_dir, "instances_train2017.json"), "data", train_images, split="train")
    copy_images_and_labels(coco_image_dir, "data", os.path.join(coco_annotation_dir, "instances_val2017.json"), "data", val_images, split="val")

    # Save the remapped categories
    save_remapped_categories(os.path.join(coco_annotation_dir, "instances_train2017.json"))

    save_data_yaml_file(os.path.join(coco_annotation_dir, "instances_train2017.json"))