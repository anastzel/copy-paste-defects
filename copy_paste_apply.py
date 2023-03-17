# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:57:56 2022

@author: anastzel
"""

from image_utils_sunlight import *
import os
import random
from tqdm import tqdm

def get_random_defect_info(number_of_defect_types):
    random_defect_type = types_of_defects[random.randint(0, number_of_defect_types-1)]

    # # This is for testing purposes
    random_defect_type = "seam"

    source_img_name = f"{random_defect_type}.jpg"
    return random_defect_type, source_img_name

def copy_paste_single_defect(random_defect_type, source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name):
    if random_defect_type in ["hole", "tearing"]:
        place_on_surface(source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name, random_defect_type)
    elif random_defect_type in ["left_border_wrinkle", "non_polished"]:
        place_on_left_border(source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name, random_defect_type)
    elif random_defect_type in ["right_border_wrinkle", "rods"]:
        place_on_right_border(source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name, random_defect_type)
    elif random_defect_type in ["upper_border_hole"]:
        place_on_upper_border(source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name, random_defect_type)
    elif random_defect_type in ["lower_border_hole", "seam"]:
        place_on_lower_border(source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name, random_defect_type)

# Type of defects
types_of_defects = ["hole", "left_border_wrinkle", "non_polished", "right_border_wrinkle", "rods", "seam", "tearing", "upper_border_hole", "lower_border_hole"]

number_of_defect_types = len(types_of_defects)

source_dir = r"C:\Users\tasos\Desktop\copy_paste\annotated_some_images_stage_1\source_imgs"
target_dir = r"C:\Users\tasos\Desktop\copy_paste\annotated_some_images_stage_1\target_imgs_1200"
generated_dir = r"C:\Users\tasos\Desktop\copy_paste\generated_images"

target_images_filenames = os.listdir(target_dir)
for i_name, filename in tqdm(enumerate(target_images_filenames[:5], start=0), total=len(target_images_filenames)):
    if filename.endswith('.jpg'):
        
        # Initial Augmentation
        #Get the defect type for the first crop and paste
        random_defect_type, source_img_name = get_random_defect_info(number_of_defect_types)

        # Define target's filename
        target_img_name = filename
        # Perform copy paste dependinf the type of the defect
        copy_paste_single_defect(random_defect_type, source_dir, source_img_name, target_dir, target_img_name, generated_dir, i_name)

        # Define the number of defects in target image
        number_of_defects = random.randint(0, 2)
        print(number_of_defects)

        # # This is for testing purposes
        number_of_defects = 0
    
        # Augment each time on the new generated image and json
        for _ in range(number_of_defects):
            #Get the defect type for the first crop and paste
            random_defect_type, source_img_name = get_random_defect_info(number_of_defect_types)
            # Define updated target's filename
            updated_target_dir = r"C:\Users\tasos\Desktop\copy_paste\generated_images"
            updated_target_img_name = rf"C:\Users\tasos\Desktop\copy_paste\generated_images\{int(i_name/2)}_defect.jpg"
            # Perform copy paste dependinf the type of the defect
            copy_paste_single_defect(random_defect_type, source_dir, source_img_name, updated_target_dir, updated_target_img_name, generated_dir, i_name)
        
