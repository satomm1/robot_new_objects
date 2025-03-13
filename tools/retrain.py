import numpy
from PIL import Image
from ultralytics import YOLO, settings
import argparse
import os

"""
This script retrains the YOLOv11 model on the updated dataset
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the YOLOv11 model on the dataset")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--pretrained-model', type=str, help='Path to the pretrained model')

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    imgsz = args.image_size
    pretrained = args.pretrained
    pretrained_model = args.pretrained_model

    print(f"Training the model for {epochs} epochs with batch size {batch_size}")
    print(f"Image size: {imgsz}")
    if pretrained:
        print(f"Using pretrained model from {pretrained_model}")
    else:
        print("Training from COCO weights")
    print("\n\n")

    # Update the datasets directory for Ultralytics
    current_dir = os.path.dirname(os.path.realpath(__file__))  # Get path to current directory
    home_dir = os.path.dirname(current_dir)  # Get the path to the ../ directory
    data_dir = os.path.join(home_dir, "updated_data")  # Path to the data directory
    settings.update({"datasets_dir": data_dir})  # Update the datasets directory

    # Remove the labels cache files
    if os.path.exists("updated_data/train/labels.cache"):
        os.remove("updated_data/train/labels.cache")
    if os.path.exists("updated_data/val/labels.cache"):
        os.remove("updated_data/val/labels.cache")    

    # Load the model
    if not pretrained:
        model = YOLO("yolo11n.pt")
    else:
        model = YOLO(pretrained_model)

    # Begin training
    model.train(data="tools/updated_data.yaml", epochs=epochs, batch=batch_size, imgsz=imgsz, freeze=18)

