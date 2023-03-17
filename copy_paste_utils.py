# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:02:00 2023

@author: anastzel
"""

import os

def get_images_with_specific_defect(dir, defect_type):
    
    defect_dir = os.path.join(dir, defect_type)
    defect_images_paths = [os.path.join(defect_dir, filename) for filename in os.listdir(defect_dir) if filename.endswith('.jpg')]

    return defect_images_paths


if __name__ == '__main__':
    
    # For debugging purposes
    source_dir = r"D:\copy_paste_pipeline\source_images"
    defect_type = "hole"
    for path in get_images_with_specific_defect(source_dir, defect_type):
        print(path)