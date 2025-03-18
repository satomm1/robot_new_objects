# A Mobile Robot Framework For Learning To Detect New Objects With Large Language Models

This is the source code for the DETC2025-164474 paper submitted to the ASME 2025 IDETC-CIE conference. In this paper, we develop a fast open set object detector for mobile robots. An LLM is used for providing unknown object names and semantics. To identify repeated unknown objects, a MobileCLIP based solution is implemented. Last, we demonstrate incremental learning by automatically saving unknown images with their labels.

This `main` branch contains code for training the open set object detector that is deployed on mobile robots.

Other git repos that are required include 
- a [ROS noetic image detection package](https://github.com/satomm1/image_detection_with_unknowns) for deploying on a mobile robot
- a [repo](https://github.com/satomm1/gemini_api) that serves as the api for the LLM queries

## Installation (Linux)
**Step 1**: Clone this repo
```
git clone https://github.com/satomm1/robot_new_objects.git
```
**Step 2**: Install the requirements
```
conda create --name detect python=3.10
conda activate detect
```
If using a GPU you must install [PyTorch](https://pytorch.org/get-started/locally/) first. Then install [Ultralytics](https://docs.ultralytics.com/quickstart/)>=8.3.80 (`pip install ultralytics`). Other Python packages you need are:
- google-generativeai (`pip install google-generativeai`)

**Step 3**: Get a [Gemini API key](https://aistudio.google.com/apikey). After you have an API key, set the key to a local environment variable:
```
export GEMINI_API_KEY=<YOUR_API_KEY>
source ~/.bashrc
```
## Downloading and Sorting Data
We use the COCO 2017 dataset (train:18GB, val:1GB). To download the data, perform the following:
```
mkdir coco && cd coco && mkdir 2017 && mkdir OSOD && cd 2017 && mkdir images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
Then, unzip the folders:
```
unzip <filename>.zip
```
Make sure the data is in the correct location (`mv val2017 images && mv train2017 images`):
```
coco/
└── 2017
    ├── images
    │   ├── val2017
    │   │   └── <val images here>
    │   └── train2017
    │       └── <train images here>
    └── annotations
        └── <annotation json files here>
```
Next, we sort the images according to objects found in the office setting. Run the `create_data.sh` script:
```
. create_data.sh
```
This file will automatically split the data and allow you to view some of the data to verify data is correctly saved. The data will be saved to the following directories:
```
data/
├── train
│   ├── images
│   └── coco_labels
├── val
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
└── categories.txt
```
The `categories.txt` files will show the mappings of object name to object id for this task.

## Produce Unknown Labels with Gemini
To use Gemini to produce extra unknown object bounding boxes, run the following python code:
```
python3 tools/get_gemini_boxes.py --start <start_num> --end <end_num> --task query
```
This code will query gemini for unknown objects and place them in the `data/train/labels_gemini_full` directory. You can use `--start` and `--end` to choose what intervals to do, and run this script concurrently to speed up the query process.

After querying, combine the coco labels with the gemini labels with:
```
python3 tools/get_gemini_boxes.py --task combine
```
This script with compare gemini boxes to coco boxes, remove any repeated boxes, and merge them into the `/data/train/labels` directory.

## Train the model
Train the OSOD model by running:
```
. train.sh
```
After training, move the model weights into the `/weights` directory with name `best.pt`. Evaluate the model by running:
```
. eval.sh
```

## Retraining the model
You should collect the data with new labels that you wish to learn. This updated data should be placed in the `/updated_data/` directory as follows:
```
updated_data/
├── train
│   ├── images
│   └── labels
├── val
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
``` 
Make sure to copy the original COCO labels from the `/data` directory too.

To retrain, update the `updated_data.yaml` file to include your new objects. Then run the retraining script:
```
. retrain.sh
```
Evaluating the new model can be done with:
```
. eval_retrained.sh
```

## Deploying the model
Take your desired model files and use them in this [repo](https://github.com/satomm1/image_detection_with_unknowns) for deployment using ROS.