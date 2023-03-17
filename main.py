# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:03:00 2023

@author: anastzel
"""

import os
import random

from tqdm import tqdm

# from image_utils_sunlight import *
from copy_paste_utils import *
from general_utils import *

# Define directories paths
source_dir = r"D:\copy_paste_pipeline\source_images"
target_dir = r"D:\copy_paste_pipeline\target_images"
output_dir = r"D:\copy_paste_pipeline\generated_images"

# Get the number of the different defect types
num_defects = len(os.listdir(source_dir))
# Get the names of the different defect types
names_defects = sorted(os.listdir(source_dir))

# For debugging purposes
# print(f"Number of different defect types: {num_defects}")
# print(f"Different defect types: {names_defects}")

target_images_filenames = get_directory_images_filenames(target_dir)

number_of_images_produced = 20

for i in tqdm(range(number_of_images_produced), total=number_of_images_produced):

    # Initial Augmentation

    # Get the defect type and the image source name for the first crop and paste
    defect_type = get_random_defect_type(names_defects)
    images_paths = get_images_with_specific_defect(source_dir, defect_type)
    source_img_name = get_source_image(images_paths)

    # Define target's filename
    target_img_name = get_random_list_item(target_images_filenames)

    copy_paste_single_defect(defect_type, source_dir, source_img_name, target_dir, target_img_name, output_dir, i)