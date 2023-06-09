# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:02:00 2023

@author: anastzel
"""

import cv2
import numpy as np
import json
import os
import math
import base64
import labelme

from PIL import Image, ImageDraw

from general_utils import *

img2json = lambda x:x.split(".")[0]+".json"
json2img = lambda x:x.split(".")[0]+".jpg"
P_2 = lambda x,y:os.path.join(x,y)
P_3 = lambda x,y,z:os.path.join(P_2(x,y),z)

def get_label_names_from_json(scan_folder, json_file):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)  
    labels = []
    for shape in label_data['shapes']:
        if not shape['label'] in labels:
            labels.append(shape['label'])
    return labels

def shape_to_mask(img_shape, points, shape_type=None, line_width=100, point_size=1):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def extract_masks_with_annotation(dimensions, scan_folder, json_file, annotation_name):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)
    instances=[]
    for shape in label_data['shapes']:
        if shape['label']==annotation_name:
            points = shape['points']           
            shape_type = shape.get('shape_type', 'polygon')
            mask = shape_to_mask(dimensions, points, shape_type) 
            instances.append(mask)
    return instances

def json_to_masks(scan_folder, json_file, dimensions, booltype=True):
    labels = get_label_names_from_json(scan_folder,json_file)
    annotation_dict = {}
    for label in labels:
        masks = extract_masks_with_annotation(dimensions, scan_folder, json_file, label)
        if not booltype:
            for i in range(0, len(masks)):
                masks[i] = bool2int(masks[i])
        annotation_dict[label]= masks
    return annotation_dict, labels

def mask_to_contours(bool_mask,max_only = True):
    tot_mask = (bool_mask*255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(tot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if max_only:
        contours = [max(contours, key = cv2.contourArea)]
    return contours

def find_centroid_contour(single_contour):
    # calculate moments for each contour
    M = cv2.moments(single_contour)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def contour_to_mask(contour, dims):
    empty_mask = (np.zeros(dims)*255).astype(np.uint8)
    tt=cv2.fillPoly(empty_mask, pts =[contour], color=(255,255,255))
    tt = tt.astype(np.bool)
    return tt

def get_contour_info(contour, dims, expand=0) :
    xpoints = [i[0][0] for i in contour]
    ypoints = [i[0][1] for i in contour]
    contour_w = max(xpoints) - min(xpoints) + 1
    contour_h = max(ypoints) - min(ypoints) + 1    
    x0 = min(xpoints) - expand*contour_w
    y0 = min(ypoints) - expand*contour_h
    xt = max(xpoints) + expand*contour_w
    yt = max(ypoints) + expand*contour_h
    center = find_centroid_contour(contour)
    contour_mask = contour_to_mask(contour, dims)
    contour_info = {}
    contour_info['center'] = center
    contour_info['x0'] = x0
    contour_info['xt'] = xt
    contour_info['y0'] = y0
    contour_info['yt'] = yt 
    contour_info['w'] = contour_w
    contour_info['h'] = contour_h
    contour_info['mask']  = contour_mask
    return contour_info  

def find_centroid_of_mask(bool_mask):
    xcoords = np.where(bool_mask)[1]
    ycoords = np.where(bool_mask)[0]
    x0 = np.min(xcoords)
    y0 = np.min(ycoords)
    xt = np.max(xcoords)
    yt = np.max(ycoords)
    w = xt - x0 + 1
    h = yt - y0 + 1
    x_center = x0 +w//2
    y_center = y0+h//2
    return (x_center, y_center),(w,h)

def translate_image(masked_image, tx, ty):
    M = np.float32([
    	[1, 0, tx],
    	[0, 1, ty]
    ])
    shifted = cv2.warpAffine(masked_image, M, (masked_image.shape[1], masked_image.shape[0]))    
    return shifted

def centralize_mask(mask):       
    s_center,dims = find_centroid_of_mask(mask.astype(bool))
    t_center = (mask.shape[1]//2, mask.shape[0]//2)
    if len(mask.shape)==3:
        mask_3channels = mask
    else:
        mask_3channels = cv2.merge((mask,mask,mask))
    tx = t_center[0] - s_center[0]
    ty = t_center[1] - s_center[1]
    mask_translated = translate_image(mask_3channels, tx, ty)
    if len(mask.shape)==3:
        return mask_translated[:,:,:]
    else:
        return mask_translated[:,:,0]
def get_mask_info(bool_mask) :
    bool_mask = bool_mask.astype(bool)
    xcoords = np.where(bool_mask)[1]
    ycoords = np.where(bool_mask)[0]
    x0 = np.min(xcoords)
    y0 = np.min(ycoords)
    xt = np.max(xcoords)
    yt = np.max(ycoords)
    w = xt - x0
    h = yt - y0
    x_center = x0 + w//2
    y_center = y0 + h//2
    surface = np.where(bool_mask>0)[0].size
    mask_info = {}
    mask_info["x0"] = x0
    mask_info["y0"] = y0
    mask_info["xt"] = xt
    mask_info["yt"] = yt
    mask_info["w"] = w
    mask_info["h"] = h
    mask_info["x_center"] = x_center
    mask_info["y_center"] = y_center
    mask_info["surface"] = surface

    return mask_info

def translate_mask(mask, mask_img=None, tx0 = 20, ty0 = 20):    
    info = get_mask_info(mask)
    x0 = info['x0']
    y0 = info['y0']
    #target to (20,20)
    # tx0 = 20
    # ty0 = 20
    if len(mask.shape)==3:
        mask_3channels = mask
    else:
        mask_3channels = cv2.merge((mask,mask,mask))
    tx = tx0 - x0
    ty = ty0 - y0
    mask_translated = translate_image(mask_3channels, tx, ty)
    if not mask_img is None:
        mask_img_translated = translate_image(mask_img, tx, ty)
        
    if len(mask.shape)==3:
        if not mask_img is None:
            return mask_translated[:,:,:] , mask_img_translated
        else:
            return mask_translated[:,:,:] 
    else:
        if not mask_img is None:
            return mask_translated[:,:,0] , mask_img_translated
        else:
            return mask_translated[:,:,0]

def adjust_mask_to_new_shape(mask,new_dims):
    #TODO: check if new mask crops the roi we need. cropping must be adjusted or the roi must move first to left corner and then crop
    if len(mask.shape)==2:
        mask = cv2.merge((mask,mask,mask))
    new_mask = np.zeros((new_dims[0], new_dims[1],3))

    # mask = mask.astype(bool)
    # mask = mask*1
    # old_surface = mask[mask>0].sum()
    h = mask.shape[0]
    w = mask.shape[1]
    new_h = new_dims[0]
    new_w = new_dims[1]
    if new_h>h and new_w>w:
        new_mask[:mask.shape[0],:mask.shape[1],:] = mask
    elif new_h<h and new_w>w:
        new_mask[:,:mask.shape[1],:] = mask[:new_h,:,:]
    elif new_h>h and new_w<w:
        new_mask[:mask.shape[0],:,:] = mask[:,:new_w,:]
    else:
        new_mask[:,:,:] = mask[:new_h,:new_w,:]
    # new_mask = new_mask*255
    # new_mask = new_mask.astype(np.uint8)
    return new_mask.astype(np.uint8)

def mask_to_mesh(mask, img_draw = None):
    temp_img = img_draw.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = [max(contours, key = cv2.contourArea)][0]
    x,y,w,h = cv2.boundingRect(contour)
    x0 = x
    xt = x + w
    y0 = y
    yt = y + h
    dividex = 30
    dividey = 9
    xr = np.linspace(x0,xt,dividex)
    yr = np.linspace(y0,yt,dividey)
    xr = np.int0(xr)
    yr = np.int0(yr)
    mesh = np.meshgrid(xr, yr)
    if not temp_img is None:
        X, Y = mesh
        positions = np.vstack([X.ravel(), Y.ravel()])
        for i in range(0,positions.shape[1]):    
            xx = positions[0][i]
            yy = positions[1][i]
            cv2.circle(temp_img, (xx,yy),0,(0,255,0),thickness=7)          
    return mesh, temp_img

def erode_mask_single(mask, kernel_dims = (40,40)):
    kernel = np.ones(kernel_dims, np.uint8) 
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    return mask_eroded

def rotate_mask(image, image_center, angle):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def contour_to_points(contour, factor = 0.005):
    epsilon = factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)     
    inst = approx.astype(np.int32).reshape(-1,2)
    points = [ list(i) for i in inst]
    points = [[int(i[0]), int(i[1])] for i in points]
    return points

def mask_to_shape(mask, factor):
    contours = mask_to_contours(mask)
    if len(contours)==0:
        return None
    # if discard_multiple_object:        
    # assert len(contours)==1, "[mask_to_shape - len(contours)>1, mask should be single object"
    contour = contours[0]
    points = contour_to_points(contour, factor)
    #points will be list with [w,h]
    return points

def resize_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height, new_width = int(scale_factor*height), int(scale_factor*width)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def get_right_border_grid(img, image_annotations, defect_type):
    # Extract the annotated regions for the source image and create a mask image for each region
    image_regions = image_annotations['shapes']
    points = image_regions[0]["points"]
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))

    # Create a 3x3 kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion on the mask
    iterations = 10
    if defect_type == "rods":
        iterations = 5

    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)

    # Apply Canny edge detection to the mask
    edges = cv2.Canny(eroded_mask, 100, 200)

    # Find the coordinates of the edge pixels
    x, y = np.nonzero(edges)

    # Find the index of the minimum value
    max_x_index = np.argmax(x)
    y_for_max_x_index = y[max_x_index]

    indexed_filtered = [i for i in range(len(y)) if y[i] >= y_for_max_x_index]

    x_final = [x[id] for id in indexed_filtered]
    y_final = [y[id] for id in indexed_filtered]
    
    return x_final, y_final

def get_left_border_grid(img, image_annotations, type_of_defect):
    # Extract the annotated regions for the source image and create a mask image for each region
    image_regions = image_annotations['shapes']
    points = image_regions[0]["points"]
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))

    # Create a 3x3 kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    if type_of_defect == "left_wrinkle":
        iterations = 30
        new_mask = cv2.erode(mask, kernel, iterations=iterations)
        # Perform erosion on the mask
    elif type_of_defect == "lag":
        iterations = 40
        new_mask = cv2.dilate(mask, kernel, iterations=iterations)


    # Apply Canny edge detection to the mask
    edges = cv2.Canny(new_mask, 100, 200)

    # Find the coordinates of the edge pixels
    x, y = np.nonzero(edges)

    # Find the index of the minimum value
    min_x_index = np.argmin(x)
    y_for_min_x_index = y[min_x_index]

    indexed_filtered = [i for i in range(len(y)) if y[i] <= y_for_min_x_index]

    x_final = [x[id] for id in indexed_filtered]
    y_final = [y[id] for id in indexed_filtered]
    
    return x_final, y_final

def get_upper_lower_border_grid(img, image_annotations, defect_type):
    # Extract the annotated regions for the source image and create a mask image for each region
    image_regions = image_annotations['shapes']
    points = image_regions[0]["points"]
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))

    # Create a 3x3 kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion on the mask
    iterations = 1
    if "upper" in defect_type:
        iterations = 20
    elif defect_type == "bottom_wrinkle":
        iterations = 10

    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)

    # Apply Canny edge detection to the mask
    edges = cv2.Canny(eroded_mask, 100, 200)

    # Find the coordinates of the edge pixels
    x, y = np.nonzero(edges)

    # Find the index of the minimum value
    min_x_index = np.argmin(x)
    max_x_index = np.argmax(x)
    y_for_min_x_index = y[min_x_index]
    y_for_max_x_index = y[max_x_index]

    indexed_filtered = [i for i in range(len(y)) if y[i] >= y_for_min_x_index and y[i] <= y_for_max_x_index]

    x_final = [x[id] for id in indexed_filtered]
    y_final = [y[id] for id in indexed_filtered]
    
    return x_final, y_final

def place_on_surface(source_dir, source_img_name, target_dir, target_img_name, output_dir, i, defect_type):
    # Get source image and json file information
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    # Get target image and json file information
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))

    # Convert source and target json files to masks
    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    # Get information about the source separator
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    # Get information about the target separator
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    # Assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    source_separator_long_edge = source_separator_info['xt'] - source_separator_info['x0']
    target_separator_long_edge = target_separator_info['xt'] - target_separator_info['x0']

    # Short edge of separator is image height
    # separator_scale_factor = target_separator_short_edge / source_separator_short_edge
    separator_scale_factor = target_separator_long_edge / source_separator_long_edge
    if separator_scale_factor > 0.8:
        additional_scale_factor = random.uniform(0.5, 0.8)
        separator_scale_factor *= additional_scale_factor
    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0] # defect should be in first position in json file
        # source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)
        
        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        source_hole_img_centered = centralize_mask(source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))

        kernel_sizes = {
            "hole": (700, 700),
            "tearing": (650, 650),
            "stamp": (1100, 1100),
            "sticker": (1100, 1100),
                        }

        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = kernel_sizes[defect_type])
  
        mesh, img_draw = mask_to_mesh(target_separator_mask, target_img)

        # Choose a random x,y pair from mesh to translate defect
        X, Y = mesh
        positions = np.vstack([X.ravel(), Y.ravel()])
        rand_i = np.random.choice(positions.shape[1], replace=False)
        tx = positions[0][rand_i]
        ty = positions[1][rand_i]       
        
        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
    
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        # Choose a random angle to rotate the defect
        rot_angles = {
            "hole": np.arange(0,360),
            "tearing": np.arange(0,45),
            "stamp": np.arange(0,45),
            "sticker": np.arange(0,360),
          
        }

        rand_angle = random.choice(rot_angles[defect_type])
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        # Define alphas for each of the defect types
        alphas = {
            "hole": 1.00,
            "tearing": 1.00,
            "stamp": 1.00,
            "sticker": 1.00,
                  }

        # Overlay the defect on the source image
        alpha = alphas[defect_type]
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(output_dir, f"{int(i)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        factors = {
            "hole": 0.005,
            "tearing": 0.001,
            "stamp": 0.005,
            "sticker": 0.005,
        }

        # Save final defective binary mask and image for debugging reasons 
        # cv2.imwrite("binary_mask_defect.jpg", rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])
        # cv2.imwrite("img_defect.jpg", rot_source_defect_img_to_top_left_rs_adj_translated)

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0], factors[defect_type])

        if defect_type == "hole":
            label = "hole"
        elif defect_type == "tearing":
            label = "tearing"
        elif defect_type == "stamp":
            label = "stamp"
        elif defect_type == "sticker":
            label = "sticker"

        defect_dict = {
      "label": label,
      "points": list_of_lists_points,
      "shape_type": "polygon",
      "flags": {}
        }

        # Open the JSON file for reading
        with open(os.path.join(target_dir, target_json_name), 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)

        img_data = labelme.LabelFile.load_image_file(new_path)
        image_data = base64.b64encode(img_data).decode('utf-8')

        # Modify the data as needed
        data['shapes'].append(defect_dict)
        data['imagePath'] = new_path
        data['imageData'] = image_data

        # Open the file for writing
        with open(f"{output_dir}/{int(i)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break # We oncly care about the first defect of the json file
    return 1

def place_on_right_border(source_dir, source_img_name, target_dir, target_img_name, output_dir, i, defect_type):
    # Get source image and json file information
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    # Get target image and json file information
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    # Convert source and target json files to masks
    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    # Get information about the source separator
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    # Get information about the target separator
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    # Assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    source_separator_long_edge = source_separator_info['xt'] - source_separator_info['x0']
    target_separator_long_edge = target_separator_info['xt'] - target_separator_info['x0']

    # Short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge
    # separator_scale_factor = target_separator_long_edge / source_separator_long_edge
    # if separator_scale_factor > 0.8:
    #     additional_scale_factor = random.uniform(0.8, 0.99)
    #     separator_scale_factor *= additional_scale_factor

    separator_scale_factor = 0.99
    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0] # defect should be in first position in json file
        # source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        source_hole_img_centered = centralize_mask(source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))

        kernel_sizes = {
            "rods": (50, 50),
            "right_wrinkle": (25, 25),
                        }

        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = kernel_sizes[defect_type])
        
        try:
            y, x = get_right_border_grid(target_img, target_image_annotations, defect_type)
            # This gives right border
            if defect_type == "right_wrinkle":
                rand_i = np.random.choice(range(int(0.3*len(x)), int(0.7*len(x))), replace=False)
            elif defect_type == "rods":
                rand_i = int(0.55*len(x))
        except ValueError:
            break

        tx, ty = x[rand_i], y[rand_i]

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))

        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        # Choose a random angle to rotate the defect
        rot_angles = {
            "rods": np.arange(-3,-2),
            "right_wrinkle": np.arange(0, 1),
        }

        rand_angle = random.choice(rot_angles[defect_type])
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        # Define alphas for each of the defect types
        alphas = {
            "rods": 1.00,
            "right_wrinkle": 1.00,
                  }

        # Overlay the defect on the source image
        alpha = alphas[defect_type]
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(output_dir, f"{int(i)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        factors = {
            "rods": 0.005,
            "right_wrinkle": 0.005,
        }

        # Save final defective binary mask and image for debugging reasons 
        # cv2.imwrite("binary_mask_defect.jpg", rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])
        # cv2.imwrite("img_defect.jpg", rot_source_defect_img_to_top_left_rs_adj_translated)

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0], factors[defect_type])

        if defect_type == "rods":
            label = "rods"
        elif defect_type == "right_wrinkle":
            label = "wrinkle"

        defect_dict = {
      "label": label,
      "points": list_of_lists_points,
      "shape_type": "polygon",
      "flags": {}
        }

        # Open the JSON file for reading
        with open(os.path.join(target_dir, target_json_name), 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)

        img_data = labelme.LabelFile.load_image_file(new_path)
        image_data = base64.b64encode(img_data).decode('utf-8')

        # Modify the data as needed
        data['shapes'].append(defect_dict)
        data['imagePath'] = new_path
        data['imageData'] = image_data

        # Open the file for writing
        with open(f"{output_dir}/{int(i)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break # We oncly care about the first defect of the json file
    return 1

def place_on_left_border(source_dir, source_img_name, target_dir, target_img_name, output_dir, i, defect_type):
    # Get source image and json file information
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    # Get target image and json file information
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    # Convert source and target json files to masks
    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    # Get information about the source separator
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    # Get information about the target separator
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    # Assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    source_separator_long_edge = source_separator_info['xt'] - source_separator_info['x0']
    target_separator_long_edge = target_separator_info['xt'] - target_separator_info['x0']

    # Short edge of separator is image height
    # separator_scale_factor = target_separator_short_edge / source_separator_short_edge
    separator_scale_factor = target_separator_long_edge / source_separator_long_edge

    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0] # defect should be in first position in json file
        # source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        source_hole_img_centered = centralize_mask(source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))

        kernel_sizes = {
            "left_wrinkle": (25, 25),
                        }

        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = kernel_sizes[defect_type])
        
        try:
            y, x = get_left_border_grid(target_img, target_image_annotations, defect_type)
            # This gives left border
            if defect_type == "left_wrinkle":
                rand_i = np.random.choice(range(int(0.3*len(x)), int(0.5*len(x))), replace=False)
        except ValueError:
            break

        tx, ty = x[rand_i], y[rand_i]

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))

        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        # Choose a random angle to rotate the defect
        rot_angles = {
            "left_wrinkle": np.arange(0, 1),
        }

        rand_angle = random.choice(rot_angles[defect_type])
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        # Define alphas for each of the defect types
        alphas = {
            "left_wrinkle": 0.80,
                  }

        # Overlay the defect on the source image
        alpha = alphas[defect_type]
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(output_dir, f"{int(i)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        factors = {
            "left_wrinkle": 0.005,
        }

        # Save final defective binary mask and image for debugging reasons 
        # cv2.imwrite("binary_mask_defect.jpg", rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])
        # cv2.imwrite("img_defect.jpg", rot_source_defect_img_to_top_left_rs_adj_translated)

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0], factors[defect_type])

        if defect_type == "left_wrinkle":
            label = "wrinkle"

        defect_dict = {
      "label": label,
      "points": list_of_lists_points,
      "shape_type": "polygon",
      "flags": {}
        }

        # Open the JSON file for reading
        with open(os.path.join(target_dir, target_json_name), 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)

        img_data = labelme.LabelFile.load_image_file(new_path)
        image_data = base64.b64encode(img_data).decode('utf-8')

        # Modify the data as needed
        data['shapes'].append(defect_dict)
        data['imagePath'] = new_path
        data['imageData'] = image_data

        # Open the file for writing
        with open(f"{output_dir}/{int(i)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break # We oncly care about the first defect of the json file
    return 1

def place_seam(source_dir, source_img_name, target_dir, target_img_name, output_dir, i, defect_type):
    # Get source image and json file information
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    # Get target image and json file information
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))

    # Convert source and target json files to masks
    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    # Get information about the source separator
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    # Get information about the target separator
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    # Assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    source_separator_long_edge = source_separator_info['xt'] - source_separator_info['x0']
    target_separator_long_edge = target_separator_info['xt'] - target_separator_info['x0']

    # Short edge of separator is image height
    # separator_scale_factor = target_separator_short_edge / source_separator_short_edge
    separator_scale_factor = target_separator_long_edge / source_separator_long_edge
    separator_scale_factor *= 1.01
    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0] # defect should be in first position in json file
        # source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)
        
        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        source_hole_img_centered = centralize_mask(source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))

        kernel_sizes = {
            "seam": (600, 600),
                        }

        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = kernel_sizes[defect_type])
  
        mesh, img_draw = mask_to_mesh(target_separator_mask, target_img)

        # Choose a random x,y pair from mesh to translate defect
        X, Y = mesh
        positions = np.vstack([X.ravel(), Y.ravel()])
        rand_i = np.random.choice(positions.shape[1], replace=False)

        sorted_x_positions = np.sort(positions[0])
        sorted_y_positions = np.sort(positions[1])

        ty = sorted_y_positions[:len(sorted_y_positions//4)][rand_i]
        tx = np.median(sorted_x_positions)       
        
        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
    
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        # Choose a random angle to rotate the defect
        rot_angles = {
            "seam": np.arange(0,1),
        }

        rand_angle = random.choice(rot_angles[defect_type])
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        # Define alphas for each of the defect types
        alphas = {
            "seam": 0.95,
                  }

        # Overlay the defect on the source image
        alpha = alphas[defect_type]
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(output_dir, f"{int(i)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        factors = {
            "seam": 0.005,
        }

        # Save final defective binary mask and image for debugging reasons 
        # cv2.imwrite("binary_mask_defect.jpg", rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])
        # cv2.imwrite("img_defect.jpg", rot_source_defect_img_to_top_left_rs_adj_translated)

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0], factors[defect_type])

        if defect_type == "seam":
            label = "seam"

        defect_dict = {
      "label": label,
      "points": list_of_lists_points,
      "shape_type": "polygon",
      "flags": {}
        }

        # Open the JSON file for reading
        with open(os.path.join(target_dir, target_json_name), 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)

        img_data = labelme.LabelFile.load_image_file(new_path)
        image_data = base64.b64encode(img_data).decode('utf-8')

        # Modify the data as needed
        data['shapes'].append(defect_dict)
        data['imagePath'] = new_path
        data['imageData'] = image_data

        # Open the file for writing
        with open(f"{output_dir}/{int(i)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break # We oncly care about the first defect of the json file
    return 1

def place_bottom_wrinkle(source_dir, source_img_name, target_dir, target_img_name, output_dir, i, defect_type):
    # Get source image and json file information
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    # Get target image and json file information
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    # Convert source and target json files to masks
    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    # Get information about the source separator
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    # Get information about the target separator
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    # Assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    source_separator_long_edge = source_separator_info['xt'] - source_separator_info['x0']
    target_separator_long_edge = target_separator_info['xt'] - target_separator_info['x0']

    # Short edge of separator is image height
    separator_scale_factor = target_separator_long_edge / source_separator_long_edge
    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0] # defect should be in first position in json file
        # source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        source_hole_img_centered = centralize_mask(source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))

        kernel_sizes = {
            "bottom_wrinkle": (50, 50),
                        }

        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = kernel_sizes[defect_type])
        
        try:
            y, x = get_upper_lower_border_grid(target_img, target_image_annotations, defect_type)
            # This gives right border
            rand_i = np.random.choice(range(int(len(x)/2), len(x)), replace=False)
        except ValueError:
            break

        tx, ty = x[rand_i], y[rand_i]

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))

        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        # Choose a random angle to rotate the defect
        rot_angles = {
            "bottom_wrinkle": np.arange(0, 1),
        }

        rand_angle = random.choice(rot_angles[defect_type])
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        # Define alphas for each of the defect types
        alphas = {
            "bottom_wrinkle": 0.95,
                  }

        # Overlay the defect on the source image
        alpha = alphas[defect_type]
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(output_dir, f"{int(i)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        factors = {
            "bottom_wrinkle": 0.005,
        }

        # Save final defective binary mask and image for debugging reasons 
        # cv2.imwrite("binary_mask_defect.jpg", rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])
        # cv2.imwrite("img_defect.jpg", rot_source_defect_img_to_top_left_rs_adj_translated)

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0], factors[defect_type])

        if defect_type == "bottom_wrinkle":
            label = "wrinkle"

        defect_dict = {
      "label": label,
      "points": list_of_lists_points,
      "shape_type": "polygon",
      "flags": {}
        }

        # Open the JSON file for reading
        with open(os.path.join(target_dir, target_json_name), 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)

        img_data = labelme.LabelFile.load_image_file(new_path)
        image_data = base64.b64encode(img_data).decode('utf-8')

        # Modify the data as needed
        data['shapes'].append(defect_dict)
        data['imagePath'] = new_path
        data['imageData'] = image_data

        # Open the file for writing
        with open(f"{output_dir}/{int(i)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break # We oncly care about the first defect of the json file
    return 1

def place_lag(source_dir, source_img_name, target_dir, target_img_name, output_dir, i, defect_type):
    # Get source image and json file information
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    # Get target image and json file information
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    # Convert source and target json files to masks
    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    # Get information about the source separator
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    # Get information about the target separator
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    # Assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    source_separator_long_edge = source_separator_info['xt'] - source_separator_info['x0']
    target_separator_long_edge = target_separator_info['xt'] - target_separator_info['x0']

    # Short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge
    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0] # defect should be in first position in json file
        # source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        source_hole_img_centered = centralize_mask(source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))

        kernel_sizes = {
            "lag": (1, 1),
                        }

        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = kernel_sizes[defect_type])
        
        try:
            y, x = get_left_border_grid(target_img, target_image_annotations, defect_type)
            # This gives left border
            if defect_type == "lag":
                rand_i = int(0.45*len(x))
        except ValueError:
            break

        tx, ty = x[rand_i], y[rand_i]

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))

        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        # Choose a random angle to rotate the defect
        rot_angles = {
            "lag": np.arange(0, 1),
        }

        rand_angle = random.choice(rot_angles[defect_type])
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        # Define alphas for each of the defect types
        alphas = {
            "lag": 1.00,
                  }

        # Overlay the defect on the source image
        alpha = alphas[defect_type]
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(output_dir, f"{int(i)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        factors = {
            "lag": 0.005,
        }

        # Save final defective binary mask and image for debugging reasons 
        # cv2.imwrite("binary_mask_defect.jpg", rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])
        # cv2.imwrite("img_defect.jpg", rot_source_defect_img_to_top_left_rs_adj_translated)

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0], factors[defect_type])

        if defect_type == "lag":
            label = "lag"

        defect_dict = {
      "label": label,
      "points": list_of_lists_points,
      "shape_type": "polygon",
      "flags": {}
        }

        # Open the JSON file for reading
        with open(os.path.join(target_dir, target_json_name), 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)

        img_data = labelme.LabelFile.load_image_file(new_path)
        image_data = base64.b64encode(img_data).decode('utf-8')

        # Modify the data as needed
        data['shapes'].append(defect_dict)
        data['imagePath'] = new_path
        data['imageData'] = image_data

        # Open the file for writing
        with open(f"{output_dir}/{int(i)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break # We only care about the first defect of the json file
    return 1

def place_defect():
    # TODO: Merge the place defect function into one
    pass

def main():
    pass

if __name__ == "__main__":
    main()