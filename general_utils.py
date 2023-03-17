# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:19:00 2023

@author: anastzel
"""

import os
import random
import numpy as np

def get_random_list_item(input_list):
    """
    Returns a random item from a list.
    """
    return random.choice(input_list)

def get_random_list_id(input_list):
    """
    Returns a random index from a list.
    """
    return random.randint(0, len(input_list) - 1)

def get_directory_images_filenames(dir):
    """
    Returns a list containing absolute paths of files inside the directory.
    """
    return [os.path.join(dir, filename) for filename in os.listdir(dir) if filename.endswith('.jpg')]

def bool2int(bool_mask):
    """
    Convert a boolean mask to a mask of integers.
    """  
    return (bool_mask*255).astype(np.uint8)

def split_directory_base_path_from_full_path(full_path):
    """
    Splits a full path into a directory and a base path.
    """ 
    list = full_path.split('\\')
    base_path = list[-1]
    directory_path = "/".join(list[:-1])

    return directory_path, base_path    

if __name__ == "__main__":

    # For debugging
    full_path = r"D:\copy_paste_pipeline\source_images\hole\hole_0.jpg"
    directory_path, base_path = split_directory_base_path_from_full_path(full_path)
    print(f"Directory path: {directory_path}")
    print(f"Base path: {base_path}")