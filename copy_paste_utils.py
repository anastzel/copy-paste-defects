# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:02:00 2023

@author: anastzel
"""

import os

from general_utils import *
from image_utils import place_on_surface, place_on_right_border, place_seam, place_bottom_wrinkle, place_on_left_border, place_lag

def get_random_defect_type(names):
    """
    Returns a string containg the name of a random defect type.
    """
    return get_random_list_item(names)

def get_images_with_specific_defect(dir, defect_type):
    """
    Returns a list containg the image paths of a specific defect type.
    """
    defect_dir = os.path.join(dir, defect_type)
    defect_images_paths = [os.path.join(defect_dir, filename) for filename in os.listdir(defect_dir) if filename.endswith('.jpg')]

    return defect_images_paths

def get_source_image(images_list):
    """
    Returns a random image path from a list of image paths.
    """
    return get_random_list_item(images_list)

# def get_source_defect_info(dir, defct_type):

def get_random_defect_info(source_dir):
    """
    Get random defect info from source directory path.
    """
    names = sorted(os.listdir(source_dir))
    # names = ["hole", "tearing", "stamp", "sticker"]
    names = ["lag"]
    
    defect_type = get_random_defect_type(names)

    images_paths = get_images_with_specific_defect(source_dir, defect_type)
    random_source_name = get_source_image(images_paths)

    return defect_type, random_source_name


def print_images_and_directory_paths(source_dir, target_dir, output_dir, source_img_name, target_img_name):
    """
    Prints directories and images paths.
    """
    print(f"\nSource directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Source image name: {source_img_name}")
    print(f"Target image name: {target_img_name}")

def copy_paste_single_defect(defect_type, source_dir, source_img_name, target_dir, target_img_name, output_dir, index):
    """
    Performs copy paste augmentation in a single image and saves the resulting image and annotation json.
    """

    if defect_type in ["hole", "tearing", "stamp", "sticker"]:
        place_on_surface(source_dir, source_img_name, target_dir, target_img_name, output_dir, index, defect_type)

    # TODO: add other defect types
    elif defect_type in ["left_wrinkle"]:
        place_on_left_border(source_dir, source_img_name, target_dir, target_img_name, output_dir, index, defect_type)
    elif defect_type in ["right_wrinkle", "rods"]:
        place_on_right_border(source_dir, source_img_name, target_dir, target_img_name, output_dir, index, defect_type)
    elif defect_type in ["seam"]:
        place_seam(source_dir, source_img_name, target_dir, target_img_name, output_dir, index, defect_type)
    elif defect_type in ["bottom_wrinkle"]:
        place_bottom_wrinkle(source_dir, source_img_name, target_dir, target_img_name, output_dir, index, defect_type)
    elif defect_type in ["lag"]:
        place_lag(source_dir, source_img_name, target_dir, target_img_name, output_dir, index, defect_type)

if __name__ == '__main__':

    # For debugging purposes
    source_dir = r"D:\copy_paste_pipeline\source_images"
    names = os.listdir(source_dir)

    defect_type = get_random_defect_type(names)
    # defect_type = "hole"

    images_paths = get_images_with_specific_defect(source_dir, defect_type)
    print(get_source_image(images_paths))