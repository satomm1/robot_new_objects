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
If using a GPU you must install [PyTorch](https://pytorch.org/get-started/locally/) first. Then install [Ultralytics](https://docs.ultralytics.com/quickstart/) (`pip install ultralytics`). Other Python packages you need are:
- google-generativeai (`pip install google-generativeai`)
- ...

**Step 3**: Get a [Gemini API key](https://aistudio.google.com/apikey). After you have an API key, set the key to a local environment variable:
```
export GEMINI_API_KEY=<YOUR_API_KEY>
source ~/.bashrc
```
