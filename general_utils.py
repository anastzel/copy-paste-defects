# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:19:00 2023

@author: anastzel
"""

import random

def get_random_list_item(input_list):
    """
    Returns a random iteam from a list.
    """

    return random.choice(input_list)

def get_random_list_id(input_list):
    """
    Returns a random index from a list.
    """
    return random.randint(0, len(input_list) - 1)

if __name__ == "__main__":

    # For debugging
    input_list = [1, 2, 3, 4, 5]
    print(get_random_list_id(input_list))
    print(get_random_list_item(input_list))