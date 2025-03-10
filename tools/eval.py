import numpy as np
from PIL import Image
from ultralytics import YOLO, settings
import argparse
import os
import tqdm

"""
This script evaluates the YOLOv11 model on the test dataset
"""

def find_tp_fn(detections, ground_truth, iou=0.5):
    """
    Find the true positive and false negative detections.
    Args:
        detections: Detected bounding boxes.
        ground_truth: Ground truth bounding boxes.
        iou: IoU threshold.
    Returns:
        true_positive: Number of true positive detections.
        false_negative: Number of false negative detections.
    """
    true_positive = 0
    false_negative = 0

    # Iterate through each ground truth box
    truth_array = np.zeros((len(ground_truth), len(detections)))
    for i in range(len(ground_truth)):
        for j in range(len(detections)):
            truth_array[i, j] = compute_iou(ground_truth[i], detections[j])

    # Match the detections to the ground truth boxes
    matched_detections = set()
    for i in range(len(ground_truth)):
        max_iou_idx = np.argmax(truth_array[i])
        max_iou = truth_array[i, max_iou_idx]
        if max_iou >= iou and max_iou_idx not in matched_detections:
            true_positive += 1
            matched_detections.add(max_iou_idx)
        else:
            false_negative += 1

    return true_positive, false_negative


def compute_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Box format is [x1, y1, x2, y2].
    Args:
        box1: First bounding box.
        box2: Second bounding box.
    Returns:
        iou: The IoU value.
    """
    # Get the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both the prediction and ground truth rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou = intersection / float(box1_area + box2_area - intersection)

    return iou

def compute_u_recall(model, test_dir):
    true_positive_u = 0
    false_negative_u = 0

    image_dir = os.path.join(test_dir, "images")
    label_dir = os.path.join(test_dir, "labels")
    image_names = os.listdir(image_dir)

    for image_name in tqdm.tqdm(image_names, desc="Processing images"):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        width, height = image.size

        # Perform detection
        results = model(image, conf=0.5, iou=0.5, device="0", visualize=False, embed=False, verbose=False)

        # Get the detections
        detections = results[0].boxes.data.cpu().numpy()
        bb = detections[:, :4]  # in (x1, y1, x2, y2) format
        conf = detections[:, 4]
        cls = detections[:, 5]

        # Only care about bounding boxes of unknown class (class 0)
        unknown_detections = bb[cls == 0]

        # Get the corresponding ground truth labels
        ground_truth_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
        with open(ground_truth_path, "r") as f:
            ground_truth = f.readlines()

        # Get the unknown ground truth boxes
        unknown_ground_truth = []
        for label in ground_truth:
            label = label.strip().split()
            class_num = int(label[0])
            if class_num != 0:
                continue

            xc = float(label[1])
            yc = float(label[2])
            w = float(label[3])
            h = float(label[4])

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height
            unknown_ground_truth.append([x1, y1, x2, y2])

        if len(unknown_detections) == 0:
            tp = 0
            fn = len(unknown_ground_truth)
        elif len(unknown_ground_truth) == 0:
            tp = 0
            fn = 0
        else:
            tp, fn = find_tp_fn(unknown_detections, unknown_ground_truth, iou=0.5)
        true_positive_u += tp
        false_negative_u += fn

    # Calculate the uRecall
    uRecall = 100 * true_positive_u / (true_positive_u + false_negative_u)
    print(f"uRecall: {uRecall:2.2f}")
    print(f"Total number of unknown objects: {true_positive_u + false_negative_u}")
    return uRecall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the YOLOv11 model on the test dataset")
    parser.add_argument('--model-name', type=str, default="runs/detect/train/weights/best.pt", help='Name of the model to evaluate')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for evaluation')

    args = parser.parse_args()
    model_name = args.model_name
    iou = args.iou

    model = YOLO(model_name)

    validation_results = model.val(data="tools/data.yaml", batch=1, device="0", iou=iou, split='test')  # Note iou is for NMS

    print(validation_results.box.map)  # mAP50-95
    print(validation_results.box.map50)  # mAP50
    print(validation_results.box.map75)  # mAP75

    # Get Unknown Recall
    u_recall = compute_u_recall(model, "data/test")