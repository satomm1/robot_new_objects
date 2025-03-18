import os
import numpy as np


cone_dir = "../updated_data/cone_data/test_labels"
cone_files = os.listdir(cone_dir)

num_cones = 0
for cone_file in cone_files:
    with open(os.path.join(cone_dir, cone_file), 'r') as f:
        lines = f.readlines()
        num_cones += len(lines)
        # print(f"{cone_file}: {num_cones} cones")

print(f"Total number of cones: {num_cones}")