# [Title]

This is the code for the [] submitted to the ASME 2025 conference. In this paper, we ...

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
## Downloading Data
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
├── 2017
│   ├── images
│   │   ├── val2017
│   │   │   └── <val images here>
│   │   └── train2017
│   │       └── <train images here>
│   └── annotations
│       └── <annotation json files here>
└── OSOD
    ├──
    | 
    └──
```
Next, we sort the images according to objects found in the office setting. Run the ___ script.