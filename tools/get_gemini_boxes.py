import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import shutil

from google import genai
from pydantic import BaseModel
from utils.nms import compute_iou
from utils.utils import create_empty_dir

# The prompt to use for the gemini query
PROMPT = ("Provide the bounding boxes for any and all objects in this image that an object detector may be interest in using the [ymin, xmin, ymax, xmax] format.")

class ObjectBB(BaseModel):
    bounding_boxes: list[float]
    object_name: str


def copy_existing(images_path, existing_path):
    """
    Copy the existing bounding boxes from the gemini dataset to the yolo dataset.
    """
    print("Copying existing bboxes given by Gemini")
    create_empty_dir("data/train/gemini_labels_full")

    existing_labels = os.listdir(existing_path)
    for image_name in tqdm(os.listdir(images_path)):
        label_name = image_name.replace(".jpg", ".txt")

        if label_name in existing_labels:
            shutil.copy(os.path.join(existing_path, label_name), "data/train/gemini_labels_full")


def get_gemini_boxes(images_path, start=0, end=None):
    image_names = os.listdir(images_path)
    gemini_label_path = "data/train/gemini_labels_full"
    gemini_label_names = os.listdir(gemini_label_path)

    print(f"Total number of images: {len(image_names)}")
    print(f"Total number of gemini labels so far: {len(gemini_label_names)}")

    if end is None:
        end = len(image_names)

    for i in tqdm(range(start, end)):
        image_name = image_names[i]
        label_name = image_name.replace(".jpg", ".txt")

        if label_name in gemini_label_names:
            continue

        # Get the path to the image and the label file
        image_path = os.path.join(images_path, image_name)

        # Load the image and get its size
        img = Image.open(image_path)
        width, height = img.size

        gemini_success = False
        while not gemini_success:
            try:
                # Ask the model to generate the bounding boxes
                response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[img, PROMPT],
                config={
                        'response_mime_type': 'application/json',
                        'response_schema': list[ObjectBB],
                        }
                )
                gemini_success = True
            except Exception as e:
                time.sleep(1)

        # Parse the response
        bboxes_and_names: list[ObjectBB] = response.parsed


        # Convert the bounding boxes to pixels and save them to a file
        all_bboxes = []
        all_names = []
        if bboxes_and_names is not None and len(bboxes_and_names) != 0:
            for bbox_and_name in bboxes_and_names:
                # Convert the bounding box to pixel coordinates
                if len(bbox_and_name.bounding_boxes) != 4:
                    continue

                y1, x1, y2, x2 = bbox_and_name.bounding_boxes
                y1 = y1/1000 * height
                x1 = x1/1000 * width
                y2 = y2/1000 * height
                x2 = x2/1000 * width

                # Get the object name
                object_name = bbox_and_name.object_name

                # Keep track of direct outputs form Gemini
                all_bboxes.append([(x1+x2)/width, (y1+y2)/height, (x2-x1)/width, (y2-y1)/height])
                all_names.append(object_name)

        # Write the gemini output to a file for later reference
        write_full_box_and_class(image_path, all_bboxes, all_names)

        if i >= end:
            return 

def write_full_box_and_class(image_path, bboxes, class_names):
    """
    Write the full bounding box and class to a file for later reference
    Does not remove any boxes or classes from ground truth or known classes
    """
    label_path = image_path.replace("images", "gemini_labels_full").replace("jpg", "txt")

    with open(label_path, "w") as f:
        for bbox, class_name in zip(bboxes, class_names):
            class_name = class_name.replace(" ", "_")
            f.write(f"{class_name} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def combine_gemini_and_coco():
    """
    Combine the gemini and coco labels into a single directory
    """
    coco_labels = "data/train/coco_labels"
    gemini_labels = "data/train/gemini_labels_full"
    combined_labels = "data/train/labels"
    images = "data/train/images"

    coco_label_names = os.listdir(coco_labels)
    gemini_label_names = os.listdir(gemini_labels)
    image_names = os.listdir(images)

    for image_name in tqdm(image_names):

        img = Image.open(os.path.join(images, image_name))
        width, height = img.size

        # Read the coco labels
        coco_label_name = image_name.replace(".jpg", ".txt")
        coco_label_path = os.path.join(coco_labels, coco_label_name)
        with open(coco_label_path, "r") as f:
            coco_labels = f.readlines() 

        # Read the gemini labels
        gemini_label_name = image_name.replace(".jpg", ".txt")
        gemini_label_path = os.path.join(gemini_labels, gemini_label_name)
        with open(gemini_label_path, "r") as f:
            gemini_labels = f.readlines()

    
        gt_bboxes = []
        for coco_label in coco_labels:
            _, xc_gt, yc_gt, w_gt, h_gt = coco_label.split()
            xc_gt = float(xc_gt)
            yc_gt = float(yc_gt)
            w_gt = float(w_gt)
            h_gt = float(h_gt)

            x1_gt = (xc_gt - w_gt/2) * width
            y1_gt = (yc_gt - h_gt/2) * height
            x2_gt = (xc_gt + w_gt/2) * width
            y2_gt = (yc_gt + h_gt/2) * height

            gt_bboxes.append([x1_gt, y1_gt, x2_gt, y2_gt])

        gemini_boxes = []
        for gemini_label in gemini_labels:
            _, xc, yc, w, h = gemini_label.split()
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)

            x1 = (xc - w/2) * width
            y1 = (yc - h/2) * height
            x2 = (xc + w/2) * width
            y2 = (yc + h/2) * height

            gemini_boxes.append([x1, y1, x2, y2])

        # Remove duplicate boxes
        keep, keep_indx = remove_duplicate_boxes_no_label(gt_bboxes, gemini_boxes)

        bboxes = np.array(keep)
        if len(bboxes) != 0:
            xc = (bboxes[:, 0] + bboxes[:, 2]) / 2 / width
            yc = (bboxes[:, 1] + bboxes[:, 3]) / 2 / height
            w = (bboxes[:, 2] - bboxes[:, 0]) / width
            h = (bboxes[:, 3] - bboxes[:, 1]) / height
            bboxes = np.stack([xc, yc, w, h], axis=1)

        with open(os.path.join(combined_labels, image_name.replace(".jpg", ".txt")), "w") as f:
            for i, bbox in enumerate(bboxes):
                f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

            # Now open original coco labels and write them to the file
            for coco_label in coco_labels:
                f.write(coco_label)

        
def remove_duplicate_boxes_no_label(correct_boxes, candidate_boxes, iou_threshold=0.7):
    keep = []
    keep_indx = []
    indx = -1
    for candidate in candidate_boxes:
        indx += 1
        is_duplicate = False
        for correct in correct_boxes:
            iou = compute_iou(candidate, correct)
            
            if iou > iou_threshold:
                is_duplicate = True
                break
            
        if not is_duplicate:
            keep.append(candidate)
            keep_indx.append(indx)

    return keep, keep_indx   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process images to generate bounding boxes.")
    parser.add_argument('--start', type=int, default=0, help='Starting index of images to process.')
    parser.add_argument('--end', type=int, default=None, help='Ending index of images to process.')
    parser.add_argument('--task', type=str, choices=['query', 'combine'], required=True, help='Task to perform: query or combine.')
    args = parser.parse_args()

    start = args.start
    end = args.end
    task = args.task

    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    client = genai.Client(api_key=gemini_api_key)

    # If have existing gemini labels, copy them to the yolo dataset
    # copy_existing("data/train/images", "../OSOD-LLM/yolo_data/OWDETR/task1/train/labels_gemini_full")

    if task == 'query':
        # Now get gemini labels for the rest of the images
        get_gemini_boxes("data/train/images", start=start, end=end)
    elif task == 'combine':
        # Combine the gemini and coco labels into a single directory
        combine_gemini_and_coco()
    