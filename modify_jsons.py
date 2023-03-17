# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:00:00 2023

@author: anastzel
"""

import os 
import json
from tqdm import tqdm

# This was created to fix the imagePath attrubute mistake in the generated json
# file after augmentation. It might not be useful anymore.

# Define the source directory containing the json annotations
src_dir = "dataset"

input_train_dir = os.path.join(src_dir, "train")
input_val_dir = os.path.join(src_dir, "val")
input_test_dir = os.path.join(src_dir, "test")

# Define the directories to run through
input_dirs = [input_train_dir, input_val_dir, input_test_dir]

for dir in input_dirs:
    # Get the list of files in the directory
    filenames = os.listdir(dir)
    for filename in tqdm(filenames, total=len(filenames)):
        # If file is a json file
        if filename.endswith('json'):
            
            # Open the JSON file
            with open(os.path.join(dir, filename), 'r') as file:
                # Read the JSON file
                data = json.load(file)

            # Update the ImagePath attribute
            data["imagePath"] = filename.replace('json', 'jpg')

            # Save the updated JSON file
            with open(os.path.join(dir, filename), 'w') as file:
                json.dump(data, file)
