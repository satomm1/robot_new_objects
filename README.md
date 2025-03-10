# [Title]

This is the code for the [] submitted to the ASME 2025 conference. In this paper, we ...

This `main` branch contains code for training models, while the `noetic` branch contains ROS code for deploying on mobile robots.

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
- ...

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

## Deploying the model
...