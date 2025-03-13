#!/bin/bash

# Run the get_data.py script
python3 tools/retrain.py --epochs 100 --batch-size 32 --pretrained --pretrained-model weights/best.pt