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
source_images_dir = r"D:\copy_paste_pipeline\source_images"
target_images_dir = r"D:\copy_paste_pipeline\target_images"
output_images_dir = r"D:\copy_paste_pipeline\generated_images"

number_of_images_to_generate = 20

# Get the number of the different defect types
num_defects = len(os.listdir(source_images_dir))
# Get the names of the different defect types
names_defects = sorted(os.listdir(source_images_dir))

target_images_filenames = get_directory_images_filenames(target_images_dir)

for i in tqdm(range(number_of_images_to_generate), total=number_of_images_to_generate):

    # Initial Augmentation

    # Get the defect type and the image source name for the first crop and paste
    defect_type = get_random_defect_type(names_defects)
    # For debugging purposes, hardcoded defect type
    defect_type = "hole"
    images_paths = get_images_with_specific_defect(source_images_dir, defect_type)
    random_source_name = get_source_image(images_paths)

    # Get random target name from available target images names
    random_target_name = get_random_list_item(target_images_filenames)

    source_dir, source_img_name = split_directory_base_path_from_full_path(random_source_name)
    target_dir, target_img_name = split_directory_base_path_from_full_path(random_target_name)
    # This is beacause of Windows, we need to remove the backslash
    output_dir = output_images_dir.replace('\\', '/')

    # For debugging purposes
    # print_images_and_directory_paths(source_dir, target_dir, output_dir, source_img_name, target_img_name)

    copy_paste_single_defect(defect_type, source_dir, source_img_name, target_dir, target_img_name, output_dir, i)
