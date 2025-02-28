import os
from PIL import Image
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
This script allows you to view the images and labels in the dataset.
"""

def draw_labels_on_image(image_path, label_path, categories):
        
        image = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        for label in labels:
            parts = label.strip().split()
            class_id, x, y, w, h = map(float, parts)
            x_min = int((x - w / 2) * image.width)
            y_min = int((y - h / 2) * image.height)
            x_max = int((x + w / 2) * image.width)
            y_max = int((y + h / 2) * image.height)
            
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min, categories[int(class_id)], color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        plt.axis('off')
        plt.savefig('tools/display.png', bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="View images and labels in the dataset")
        parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True, help='Directory containing images')
        return parser.parse_args()

    args = parse_args()
    split = args.split


    image_dir = f"data/{split}/images"
    label_dir = f"data/{split}/labels"

    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    # Read the categories
    with open("data/categories.txt", "r") as f:
        categories = f.readlines()
        categories = [category.strip().split() for category in categories]
        categories = {int(category[0]): ' '.join(category[1:]) for category in categories}

    while True:
        image_file = random.choice(image_files)
        label_file = image_file.replace('.jpg', '.txt')

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        draw_labels_on_image(image_path, label_path, categories)

        user_input = input(f"Saved image {image_file} to tools/display.png. Press 'q' to quit or any other key to view another image: ")
        if user_input.lower() == 'q':
            break