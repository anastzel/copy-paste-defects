# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:19:00 2023

@author: anastzel
"""

import os
import random

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

if __name__ == "__main__":

    # For debugging
    input_list = [1, 2, 3, 4, 5]
    print(get_random_list_id(input_list))
    print(get_random_list_item(input_list))