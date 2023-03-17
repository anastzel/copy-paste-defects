# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:02:00 2023

@author: anastzel
"""

import os

from general_utils import *

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

if __name__ == '__main__':

    # For debugging purposes
    source_dir = r"D:\copy_paste_pipeline\source_images"
    names = os.listdir(source_dir)

    defect_type = get_random_defect_type(names)
    # defect_type = "hole"

    images_paths = get_images_with_specific_defect(source_dir, defect_type)
    print(get_source_image(images_paths))