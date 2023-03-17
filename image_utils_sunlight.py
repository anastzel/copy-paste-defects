# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:38:40 2021

@author: lleontar
"""

import cv2
import numpy as np
import json
import math
import os
import random
from PIL import Image, ImageDraw
import copy
from tqdm import tqdm
import base64
import labelme

img2json = lambda x:x.split(".")[0]+".json"
json2img = lambda x:x.split(".")[0]+".jpg"
P_2 = lambda x,y:os.path.join(x,y)
P_3 = lambda x,y,z:os.path.join(P_2(x,y),z)

def show(img, downsize):
    img = cv2.resize(img, (img.shape[1]//downsize, img.shape[0]//downsize))
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def bool2int(bool_mask):
        return (bool_mask*255).astype(np.uint8)
    
def int2bool(int_mask):
    return int_mask.astype(np.bool)
    
def mask_to_contours(bool_mask,max_only = True):
    tot_mask = (bool_mask*255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(tot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if max_only:
        contours = [max(contours, key = cv2.contourArea)]
    return contours

def contour_to_mask(contour, dims):
    empty_mask = (np.zeros(dims)*255).astype(np.uint8)
    tt=cv2.fillPoly(empty_mask, pts =[contour], color=(255,255,255))
    tt = tt.astype(np.bool)
    return tt

def split_mask_to_masks(mask, boolmask=True):
    dims = mask.shape[:2]
    split_masks = []
    tot_mask = (mask*255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(tot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        empty_mask = (np.zeros(dims)*255).astype(np.uint8)
        tt=cv2.fillPoly(empty_mask, pts =[contour], color=(255,255,255))
        if boolmask:
            split_masks.append(tt.astype(np.bool))
        else:
            split_masks.append(bool2int(tt))
    return split_masks

def find_centroid_contour(single_contour):
    # calculate moments for each contour
    M = cv2.moments(single_contour)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

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
    
def rotate_mask(image, image_center, angle):
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def merge_instances(dimensions,instances):
    no_instances = instances.__len__()
    total_mask = np.zeros(dimensions).astype(bool)
    for i in range(0, no_instances):
        predicted_mask = instances[i]
        total_mask = np.logical_or(total_mask, predicted_mask)
    return total_mask 
   
def merge_mask_dict_per_category(mask_dict,dims, categories):
    #first merge instance per category to handle annotations of same region but different polygons
    #then split the instances to retrieve separate instances for each category
    merged_mask_dict={}
    for category in categories:         
        merged_mask_dict[category] = []       
    for key, value in mask_dict.items():
        total_category_mask = merge_instances(dims, value)
        split_masks = split_mask_to_masks(total_category_mask, False)
        for split_mask in split_masks:
            merged_mask_dict[key].append(split_mask)
    return merged_mask_dict
              
def dilate_mask(whatevers_mask):
    dims = whatevers_mask.shape[:2]
    whatevers_img = (whatevers_mask*255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(whatevers_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((40,40), np.uint8) 
    split_masks=[]
    split_masks_dilate = []
    for contour in contours:
        empty_mask = (np.zeros(dims)*255).astype(np.uint8)
        tt=cv2.fillPoly(empty_mask, pts =[contour], color=(255,255,255))
        split_masks.append(tt.astype(np.bool))
    for ii,s in enumerate(split_masks):
        s_img = (s*255).astype(np.uint8)
        s_dilate = cv2.dilate(s_img, kernel, iterations=1)
        split_masks_dilate.append(s_dilate.astype(np.bool))
    whatevers_mask = merge_instances(dims,split_masks_dilate)
    return whatevers_mask
  
def erode_mask(whatevers_mask,kernel_dims = (40,40)):
    dims = whatevers_mask.shape[:2]
    whatevers_img = (whatevers_mask*255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(whatevers_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones(kernel_dims, np.uint8) 
    split_masks=[]
    split_masks_erode = []
    for contour in contours:
        empty_mask = (np.zeros(dims)*255).astype(np.uint8)
        tt=cv2.fillPoly(empty_mask, pts =[contour], color=(255,255,255))
        split_masks.append(tt.astype(np.bool))
    for ii,s in enumerate(split_masks):
        s_img = (s*255).astype(np.uint8)
        s_dilate = cv2.erode(s_img, kernel, iterations=1)
        split_masks_erode.append(s_dilate.astype(np.bool))
    whatevers_mask = merge_instances(dims,split_masks_erode)
    return whatevers_mask
  
def erode_mask_single(mask,kernel_dims = (40,40)):
    kernel = np.ones(kernel_dims, np.uint8) 
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    return mask_eroded
 
  
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

def mask_to_shape(mask):
    contours = mask_to_contours(mask)
    if len(contours)==0:
        return None
    # if discard_multiple_object:        
    # assert len(contours)==1, "[mask_to_shape - len(contours)>1, mask should be single object"
    contour = contours[0]
    points = contour_to_points(contour)
    #points will be list with [w,h]
    return points
    
    
def contour_to_points(contour):
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)     
    inst = approx.astype(np.int32).reshape(-1,2)
    points = [ list(i) for i in inst]
    points = [[int(i[0]), int(i[1])] for i in points]
    return points

    
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

def masks_dict_to_labelme_dict(annotation_dict, dims_h_w, labels, json_name):
    add_shapes = []
    labelme_empty_json = get_empty_json_format(dims_h_w)
    for key, arrlist in annotation_dict.items():
      for i in range(0,len(arrlist)):
          points = mask_to_shape(annotation_dict[key][i])
          if points==None:
              continue
          shape = {}
          shape["label"] = key
          shape["shape_type"] ="polygon"    
          shape["points"] = points
          add_shapes.append(shape)

    # for label in labels:
    #     for i in range(0, len(annotation_dict[label])):
    #         points = mask_to_shape(annotation_dict[label][i])
    #         shape = {}
    #         shape["label"] = label#+"_detection"
    #         shape["shape_type"] ="polygon"    
    #         shape["points"] = points
    #         add_shapes.append(shape)
    labelme_empty_json['shapes'] = add_shapes
    labelme_empty_json["imageHeight"]= dims_h_w[0]
    labelme_empty_json["imageWidth"]= dims_h_w[1]
    labelme_empty_json["imagePath"]= json2img(json_name)
    return labelme_empty_json
    # with open(P_2(scan_folder, json_file),'w') as f:
    #     json.dump(annotation_dict, f, indent=2)  
    
def get_label_names_from_json(scan_folder, json_file):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)  
    labels = []
    for shape in label_data['shapes']:
        if not shape['label'] in labels:
            labels.append(shape['label'])
    return labels

def get_label_names_from_multiple_jsons(jsons_path_list):
    discover_labels=[]
    for json_i in jsons_path_list:
        folder = "\\".join(json_i.split("\\")[:-1])
        json_name = json_i.split("\\")[-1]
        found_labels = get_label_names_from_json(folder, json_name)
        keep_labels = [i for i in found_labels if i not in discover_labels]
        discover_labels.extend(keep_labels)
    return discover_labels
    
    
def get_dims_from_json(scan_folder, json_file):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f) 
    h = label_data["imageHeight"]
    w = label_data["imageWidth"]
    return (h,w)

def has_dict_these_categories(dictionary, categories):
    found = False
    for shape in dictionary['shapes']:
        assert not shape['points'][0][0] is None, " error in has_json_these_categories"
        if shape['label'] in categories:
            found = True
    return found

def has_json_these_categories(scan_folder, json_file, categories):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)
    found = False
    count=0
    for shape in label_data['shapes']:
        assert not shape['points'][0][0] is None, " error in has_json_these_categories"
        if shape['label'] in categories:
            found = True
            count+=1
    return found, count



def get_empty_json_format(dimensions_h_w):
    dict1 = {}
    Pixels = {"x": 0, "y": 0}
    dict1["Pixels"] = Pixels  
    dict1["imageData"] = None
    dict1["imageHeight"]= dimensions_h_w[0]
    dict1["imageWidth"]= dimensions_h_w[1]
    dict1["imagePath"]= ""
    shapes = []
    dict1["shapes"] = shapes   
    return dict1
    
def get_dict_from_json(scan_folder, json_file):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)  
    return label_data

def save_dict_to_json(scan_folder, json_file, dict):
    with open(P_2(scan_folder,json_file),'w') as f:
        json.dump(dict, f, indent=2)

def is_json_good(scan_folder, json_file, dims, thres=0.99):
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)
    for shape in label_data['shapes']:
        if shape['points'][0][0]==None:
             continue
        if len(shape['points'])<=2:
            print("found shape with less than 3 points")
            return False
        if shape['points'][0][0]>dims[1] or shape['points'][0][1]>dims[0]:
            print("points coords are greater than image limits")
            return False
        shape_type = shape.get('shape_type', 'polygon')
        mask = shape_to_mask(dims, shape['points'], shape_type)
        surface = np.where(mask==True)[0].size
        if surface>thres*dims[0]*dims[1]:
            print("json file: {}, surface of defect: {} is greather than {}*surface of whole image".format(json_file, shape['label'], thres))
            return False
    return True
        
            

def keep_only_categories_in_json(scan_folder, json_file, dims, categories):
    assert is_json_good(scan_folder, json_file, dims), "json file: {} not good".format(json_file)
    with open(P_2(scan_folder, json_file)) as f:
        label_data = json.load(f)
    shape_new = []
    for shape in label_data['shapes']:
        if shape['points'][0][0]==None:
             continue
        if shape['label'] in categories:
            shape_new.append(shape)
    label_data['shapes'] = shape_new
    with open(P_2(scan_folder, json_file),'w') as f:
         json.dump(label_data, f, indent=2)
      
def is_mask_good(mask):
    contours = mask_to_contours(mask)
    assert len(contours)==1, "is_mask_good: contours is >1 but mask should be single object"
    contour= contours[0]
    if len(contour)<=2:
        return False
    else:
        return True 
    
def get_only_good_masks(masks):
    filtered_masks = []
    for mask in masks:
        if is_mask_good(mask):
            filtered_masks.append(mask)
    return filtered_masks
    
def translate_image(masked_image, tx, ty):
    M = np.float32([
    	[1, 0, tx],
    	[0, 1, ty]
    ])
    shifted = cv2.warpAffine(masked_image, M, (masked_image.shape[1], masked_image.shape[0]))    
    return shifted
 
def is_fully_overlapped(source_mask, target_mask):
    result = np.logical_and(source_mask, target_mask)
    intersection_pixels = np.sum(result>0)
    source_mask_pixels = np.sum(source_mask>0)
    if source_mask_pixels==intersection_pixels:
        full_overlap=True
    else:
        full_overlap = False
    return full_overlap    
    
def translation_inside_region(limit_to_this_region_mask, mask, masked_image, only_full_overlap = True):
    contours = mask_to_contours(mask)
    one_contour = contours[0]
    x,y,w,h = cv2.boundingRect(one_contour)

    #make a random movement in x form list of possible x coords
    ycoords = np.where(limit_to_this_region_mask)[0]
    xcoords = np.where(limit_to_this_region_mask)[1]
    
    #make random choice for x,y for the center
    targety,targetx = random.choice(ycoords), random.choice(xcoords)
    tx = targetx - x
    ty = targety - y   
    shifted_mask = translate_image(bool2int(mask), tx, ty)
    shifted_bool_mask = int2bool(shifted_mask) 
    while not is_fully_overlapped(shifted_mask, limit_to_this_region_mask):
        targety,targetx = random.choice(ycoords), random.choice(xcoords)
        tx = targetx - x
        ty = targety - y
        shifted_mask = translate_image(bool2int(mask), tx, ty)
        shifted_bool_mask = int2bool(shifted_mask)
    shifted_masked_image = translate_image(masked_image, tx, ty)
    return shifted_bool_mask, shifted_masked_image
    #check if fully overlapped

def alpha_blend_two_images(foreground, foreground_alpha_mask, background):
    foreground = foreground
    background = background
    alpha= cv2.cvtColor(foreground_alpha_mask, cv2.COLOR_GRAY2BGR)
    
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    return outImage
#     # shape = [(cy, cx), (w, h)]  
#     # topleft = (x - w//2, y - h//2)
#     # rightbottom = (x + w//2 , y + h//2)
#     # mask = np.zeros(mask.shape[:2], dtype=np.uint8)
#     # mask = Image.fromarray(mask)
#     # draw = ImageDraw.Draw(mask)
#     # draw.rectangle(shape, outline=1, fill=1)
#     return shifted

def crop_mask(mask, crop_info):
    x0 = crop_info[0]
    y0 = crop_info[1]
    windowx = crop_info[2]
    windowy = crop_info[3]
    cropped = mask[y0:y0+windowy, x0:x0+windowx]       
    return cropped

def crop_mask_dict(mask_dict, crop_info):
    #crop info is (x0,y0,windowx,windowy), x0,y0 is top left
    x0 = crop_info[0]
    y0 = crop_info[1]
    windowx = crop_info[2]
    windowy = crop_info[3]
    
    new_mask_dict = {}
    for key, arrlist in mask_dict.items():
        new_mask_dict[key]=[]
        for i in range(0,len(arrlist)):
            cropped = mask_dict[key][i][y0:y0+windowy, x0:x0+windowx]       
            if np.sum(cropped[cropped>0])>0:
                new_mask_dict[key].append(cropped)         
    return new_mask_dict

def crop_mask_dict_with_instance_ids(mask_dict, crop_info, labels_of_interest = ['defect'], required_residual = 0.5):
    #crop info is (x0,y0,windowx,windowy), x0,y0 is top left
    x0 = crop_info[0]
    y0 = crop_info[1]
    windowx = crop_info[2]
    windowy = crop_info[3]
    mask_dict_with_ids = copy.deepcopy(mask_dict)
    idx=1
    for key, arrlist in mask_dict_with_ids.items():
        if not key in labels_of_interest:
            #background is '0'
            for i in range(0,len(arrlist)):
                temp = (mask_dict_with_ids[key][i]/255) * 0
                mask_dict_with_ids[key][i] = temp
        else:
            for i in range(0,len(arrlist)):
                temp = (mask_dict_with_ids[key][i]/255) * idx
                mask_dict_with_ids[key][i] = temp
    #mask_dict_with_ids has ids only for labels_of_interest with idx>=1. else ids are 0 (background or not in labels_of_interst)
    new_mask_dict = {}
    #check if at least one cropped instance is big enough
    check=[]
    crop_has_defect = False
    for key, arrlist in mask_dict_with_ids.items():
        new_mask_dict[key]=[]
        for i in range(0,len(arrlist)):
            current_instance_array  = mask_dict_with_ids[key][i]
            current_instance_id = np.unique(current_instance_array)
            if current_instance_id.shape[0]<2 and 0 in current_instance_id:
                #only '0' in array therefore only background in this instance
                continue
            #exclude 0 idx which is background
            current_defect_instance_id=int(current_instance_id[current_instance_id>0][0])
            old_surface_pixels = np.sum(current_instance_array[current_instance_array==current_defect_instance_id])
            cropped_array = current_instance_array[y0:y0+windowy, x0:x0+windowx]    
            new_surface_pixels = np.sum(cropped_array[cropped_array==current_defect_instance_id])
            #check how big roi pixels are
            if new_surface_pixels>required_residual*old_surface_pixels:
                new_mask_dict[key].append(cropped_array)         
                check.append(True)

    if np.any(check):
        crop_has_defect=True
    return new_mask_dict, crop_has_defect
    
def copy_paste(source_mask, target_image, target_mask, output_folder, save = False, dilation = False, erosion = False, rotation = True):
#with random rotate and translate fully overlapped
#first translate and then random rotate
#1 get mask
#2 bitwise image and mask
#3 translate masked_image
#4 get mask contour
#5 rotate translated masked_image
    if dilation:
        source_mask = dilate_mask(source_mask)
    if erosion:
        source_mask = erode_mask(source_mask)
    source_image_masked = cv2.bitwise_and(target_image,target_image, mask = bool2int(source_mask))
    shifted_source_mask, shifted_source_image_masked = translation_inside_region(target_mask, source_mask, source_image_masked)
    cv2.imwrite(P_2(output_folder,"shifted_source_mask.png"), bool2int(shifted_source_mask))
    cv2.imwrite(P_2(output_folder, "shifted_source_image_masked.png"), shifted_source_image_masked)
    transformed_mask = bool2int(shifted_source_mask)
    transformed_image_masked = shifted_source_image_masked
    if rotation:
        angles = np.arange(0,360)
        rand_angle = random.choice(angles)
        contours = mask_to_contours(shifted_source_mask)     
        one_contour = contours[0]
        cx,cy = find_centroid_contour(one_contour)
        rot_shifted_source_mask =  rotate_mask(transformed_mask, (cx,cy), rand_angle)
        rot_shifted_source_image_masked =  rotate_mask(transformed_image_masked, (cx,cy), rand_angle)
        cv2.imwrite(P_2(output_folder,"rot_shifted_source_mask.png"), rot_shifted_source_mask)
        cv2.imwrite(P_2(output_folder,"rot_shifted_source_image_masked.png"), rot_shifted_source_image_masked)
        transformed_mask = rot_shifted_source_mask
        transformed_image_masked = rot_shifted_source_image_masked
  
    #paste transformed defect in image
    transformed_image_masked = cv2.GaussianBlur(transformed_image_masked,(5,5),0)
    final_img = alpha_blend_two_images(transformed_image_masked, transformed_mask, target_image)
    return final_img, transformed_mask

def get_upper_lower_border_grid(img, image_annotations, type_of_defect):
    # Extract the annotated regions for the source image and create a mask image for each region
    image_regions = image_annotations['shapes']
    points = image_regions[0]["points"]
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))

    # Create a 3x3 kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion on the mask
    iterations = 1
    if "upper" in type_of_defect:
        iterations = 20
    elif type_of_defect == "seam":
        iterations = 20

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

def get_left_border_grid(img, image_annotations, type_of_defect):
    # Extract the annotated regions for the source image and create a mask image for each region
    image_regions = image_annotations['shapes']
    points = image_regions[0]["points"]
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))

    # Create a 3x3 kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    if type_of_defect == "left_border_wrinkle":
        iterations = 20
        # Perform erosion on the mask
        new_mask = cv2.erode(mask, kernel, iterations=iterations)
    elif type_of_defect == "non_polished":
        iterations = 55
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

def get_right_border_grid(img, image_annotations, type_of_defect):
    # Extract the annotated regions for the source image and create a mask image for each region
    image_regions = image_annotations['shapes']
    points = image_regions[0]["points"]
    mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))

    # Create a 3x3 kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion on the mask
    iterations = 20
    if type_of_defect == "rods":
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

def place_on_right_border(source_dir, source_img_name, target_dir, target_img_name, defect_dir, i, type_of_defect):

    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    # source_img_copy = source_img.copy()
    source_json_name = source_img_name.replace((".jpg"), (".json"))
    with open(P_2(source_dir, source_json_name), 'r') as f:
        source_image_annotations = json.load(f)

    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_img_copy = target_img.copy()
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    #assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    #short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge

    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0]

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))
        
        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = (50,50))

    
        try:
            y, x = get_right_border_grid(target_img, target_image_annotations, type_of_defect)
            # This gives upper border
            if type_of_defect == "right_border_wrinkle":
                rand_i = np.random.choice(range(int(0.2*len(x)), int(0.8*len(x))), replace=False)
            elif type_of_defect == "rods":
                rand_i = int(0.5*len(x))
        except ValueError:
            break

        tx, ty = x[rand_i], y[rand_i]

        # middle_x, middle_y = x[int(3*len(x)/4)], y[int(3*len(y)/4)]
        # cv2.circle(target_img_copy, (middle_x, middle_y), 10, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        # for i in range(0, len(x)):
        #     cv2.circle(target_img_copy, (y[i], x[i]), 5, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        angle = 0
        if type_of_defect == "rods":
            angle = -3
        
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, angle)

        alpha = 0.85
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(defect_dir, f"{int(i/2)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])

        if type_of_defect == "right_border_wrinkle":
            label = "wrinkle"
        elif type_of_defect == "rods":
            label = "rods"

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
        with open(f"{defect_dir}/{int(i/2)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)

        break

def place_on_left_border(source_dir, source_img_name, target_dir, target_img_name, defect_dir, i, type_of_defect):

    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    # source_img_copy = source_img.copy()
    source_json_name = source_img_name.replace((".jpg"), (".json"))
    with open(P_2(source_dir, source_json_name), 'r') as f:
        source_image_annotations = json.load(f)

    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_img_copy = target_img.copy()
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    #assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    #short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge

    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0]

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))
        
        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = (50,50))

        try:
            y, x = get_left_border_grid(target_img, target_image_annotations, type_of_defect)
            # This gives left border
            if type_of_defect == "left_border_wrinkle":
                rand_i = np.random.choice(range(int(0.3*len(x)), int(0.5*len(x))), replace=False)
            elif type_of_defect == "non_polished":
                rand_i = int(5.8*len(x)/10)
        except ValueError:
            break

        tx, ty = x[rand_i], y[rand_i]

        # middle_x, middle_y = x[int(3*len(x)/4)], y[int(3*len(y)/4)]
        # cv2.circle(target_img_copy, (middle_x, middle_y), 10, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        # for i in range(0, len(x)):
        #     cv2.circle(target_img_copy, (y[i], x[i]), 5, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        if type_of_defect == "left_border_wrinkle":
            angle = 0
        elif type_of_defect == "non_polished":
            angle = 0

        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, angle)


        alpha = 0.85
        if type_of_defect == "non_polished":
            alpha  = 0.95
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(defect_dir, f"{int(i/2)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])

        if type_of_defect == "left_border_wrinkle":
            label = "wrinkle"
        elif type_of_defect == "non_polished":
            label = "non_polished"

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
        with open(f"{defect_dir}/{int(i/2)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break

def place_on_upper_border(source_dir, source_img_name, target_dir, target_img_name, defect_dir, i, type_of_defect):

    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    # source_img_copy = source_img.copy()
    source_json_name = source_img_name.replace((".jpg"), (".json"))
    with open(P_2(source_dir, source_json_name), 'r') as f:
        source_image_annotations = json.load(f)

    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_img_copy = target_img.copy()
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    #assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    #short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge

    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0]

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))
        
        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = (50,50))
     
        y, x = get_upper_lower_border_grid(target_img, target_image_annotations, type_of_defect)

        # This gives upper border
        rand_i = np.random.choice(range(int(len(x)/2)), replace=False)


        tx, ty = x[rand_i], y[rand_i]

        # middle_x, middle_y = x[int(3*len(x)/4)], y[int(3*len(y)/4)]
        # cv2.circle(target_img_copy, (middle_x, middle_y), 10, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        # for i in range(0, len(x)):
        #     cv2.circle(target_img_copy, (y[i], x[i]), 5, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, 0)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, 0)

        alpha = 0.80
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(defect_dir, f"{int(i/2)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])

        defect_dict = {
      "label": "hole",
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
        with open(f"{defect_dir}/{int(i/2)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)

        break

def place_on_lower_border(source_dir, source_img_name, target_dir, target_img_name, defect_dir, i, type_of_defect):

    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    # source_img_copy = source_img.copy()
    source_json_name = source_img_name.replace((".jpg"), (".json"))
    with open(P_2(source_dir, source_json_name), 'r') as f:
        source_image_annotations = json.load(f)

    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_img_copy = target_img.copy()
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)

    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    #assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    #short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge

    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0]

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))
        
        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = (50,50))
     
        y, x = get_upper_lower_border_grid(target_img, target_image_annotations, type_of_defect)

        # This gives lower border
        rand_i = np.random.choice(range(int(len(x)/2), len(x)), replace=False)

        if "seam" in type_of_defect:
            rand_i = int(7.35*len(x)/10)


        tx, ty = x[rand_i], y[rand_i]

        # middle_x, middle_y = x[int(3*len(x)/4)], y[int(3*len(y)/4)]
        # cv2.circle(target_img_copy, (middle_x, middle_y), 10, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        # for i in range(0, len(x)):
        #     cv2.circle(target_img_copy, (y[i], x[i]), 5, (0, 255, 255), -1)
        # cv2.imwrite("target_img_copy.jpg", target_img_copy)

        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        angle = 0
        if type_of_defect == "seam":
            angle = -4

        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, angle)

        alpha = 0.80
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(defect_dir, f"{int(i/2)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])

        if type_of_defect == "lower_border_hole":
            label = "hole"
        elif type_of_defect == "seam":
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
        with open(f"{defect_dir}/{int(i/2)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)

        break
    
def place_on_metal(source_dir, source_img_name, target_dir, target_img_name, defect_dir, i, type_of_defect):
    
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_json_name = source_img_name.replace((".jpg"), (".json"))
    # Load the JSON file containing the annotated regions for the source image
    with open(P_2(source_dir, source_json_name), 'r') as f:
        source_image_annotations = json.load(f)
    
    # Extract the annotated regions for the source image and create a mask image for each region
    source_image_regions = source_image_annotations['shapes']
    source_points = source_image_regions[-1]["points"]
    source_mask = np.zeros_like(source_img[:, :, 0])
    cv2.fillPoly(source_mask, np.int32([source_points]), (255, 255, 255))
    
    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_img_copy = target_img.copy()
    target_json_name = target_img_name.replace((".jpg"), (".json"))
    # Load the JSON file containing the annotated regions for the target image
    with open(P_2(target_dir, target_json_name), 'r') as f:
        target_image_annotations = json.load(f)
    
    # Extract the annotated regions for the target image and create a mask image for each region
    target_image_regions = target_image_annotations['shapes']
    if len(target_image_regions) > 1:
        target_points = target_image_regions[1]["points"]
        target_mask = np.zeros_like(target_img[:, :, 0])
        cv2.fillPoly(target_mask, np.int32([target_points]), (255, 255, 255))

        # Find the coordinates of the non-black pixels for source image
        source_coords = cv2.findNonZero(source_mask)
        # Get the bounding box of the non-black regions
        x_s, y_s, w_s, h_s = cv2.boundingRect(source_coords)
        # Crop the image using the bounding box coordinates
        cropped_img_source = source_img[y_s:y_s+h_s, x_s:x_s+w_s]

        # Find the coordinates of the non-black pixels for target image
        target_coords = cv2.findNonZero(target_mask)
        # Get the bounding box of the non-black regions
        x_t, y_t, w_t, h_t = cv2.boundingRect(target_coords)
        # Crop the image using the bounding box coordinates
        cropped_img_target = target_img[y_t:y_t+h_t, x_t:x_t+w_t]

        cropped_img_source_resized = cv2.resize(cropped_img_source, (cropped_img_target.shape[1], cropped_img_target.shape[0]))
        target_img_copy[y_t:y_t+h_t, x_t:x_t+w_t] = cropped_img_source_resized

        new_path = P_2(defect_dir, f"{type_of_defect}_{int(i/2)}.jpg")
        cv2.imwrite(new_path, target_img_copy.astype(np.uint8))
    else:
        pass

    return 1


def place_on_surface(source_dir, source_img_name, target_dir, target_img_name, defect_dir, i, type_of_defect):
    source_img = cv2.imread(P_2(source_dir, source_img_name))
    source_dims = source_img.shape[:2]
    source_json_name = source_img_name.replace((".jpg"), (".json"))

    target_img = cv2.imread(P_2(target_dir, target_img_name))
    target_dims = target_img.shape[:2]
    target_json_name = target_img_name.replace((".jpg"), (".json"))

    source_masks_dict, _ = json_to_masks(source_dir, source_json_name, source_dims, booltype=False)
    target_masks_dict, _ = json_to_masks(target_dir, target_json_name, target_dims, booltype=False)
    
    source_separator_mask = source_masks_dict['separator'][0]
    source_separator_contour = mask_to_contours(source_separator_mask)[0]
    source_separator_info = get_contour_info(source_separator_contour, source_dims)
    
    target_separator_mask = target_masks_dict['separator'][0]
    target_separator_contour = mask_to_contours(target_separator_mask)[0]
    target_separator_info = get_contour_info(target_separator_contour, target_dims)

    #assumes separator is in horizontal orientation in image
    source_separator_short_edge = source_separator_info['yt'] - source_separator_info['y0']
    target_separator_short_edge = target_separator_info['yt'] - target_separator_info['y0']

    #short edge of separator is image height
    separator_scale_factor = target_separator_short_edge / source_separator_short_edge
    
    for key, _ in source_masks_dict.items():
        source_defect_mask = source_masks_dict[key][0]
        ## source_hole_mask_centered = centralize_mask(source_hole_mask)

        source_defect_img =  cv2.bitwise_and(source_img, source_img, mask = bool2int(source_defect_mask))

        source_defect_mask_to_top_left, source_defect_img_to_top_left = translate_mask(source_defect_mask, source_defect_img)
        # cv2.imwrite("source_hole_mask_to_top_left.png", source_defect_mask_to_top_left)
        # cv2.imwrite("source_hole_img_to_top_left.png", source_defect_img_to_top_left)
        
        source_defect_contour = mask_to_contours(source_defect_mask)[0]
        source_defect_info = get_contour_info(source_defect_contour, source_dims)

        # cv2.imwrite("source_hole_img.png", source_defect_img)
        source_hole_img_centered = centralize_mask(source_defect_img)
        # cv2.imwrite("source_hole_img_centered.png", source_hole_img_centered)

        ww = source_defect_img.shape[1]
        hh = source_defect_img.shape[0]
        source_defect_mask_to_top_left_rs = cv2.resize(source_defect_mask_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        # cv2.imwrite("source_hole_mask_to_top_left_rs.png",source_defect_mask_to_top_left_rs)   
        source_defect_img_to_top_left_rs = cv2.resize(source_defect_img_to_top_left, (int(ww*separator_scale_factor), int(hh*separator_scale_factor)))
        # cv2.imwrite("source_hole_img_to_top_left_rs.png", source_defect_img_to_top_left_rs)   
        
        # source_hole_img_rs = cv2.resize(source_hole_img_centered, (int(ww*separator_scale_factor),int(hh*separator_scale_factor)))
        # source_hole_img_centered_rescaled_to_separator = cv2.resize(source_hole_img_centered, (int(ww*separator_scale_factor),int(hh*separator_scale_factor)))
        # cv2.imwrite("source_hole_img_centered_rescaled_to_separator.png",source_hole_img_rs)        
        source_defect_mask_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_mask_to_top_left_rs, target_dims)
        # cv2.imwrite("source_hole_mask_to_top_left_rs_adj.png", source_hole_mask_to_top_left_rs_adj)   
        source_defect_img_to_top_left_rs_adj = adjust_mask_to_new_shape(source_defect_img_to_top_left_rs, target_dims)
        # cv2.imwrite("source_hole_img_to_top_left_rs_adj.png", source_hole_img_to_top_left_rs_adj)  
        
        (tx_, ty_), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj.astype(bool))
        # #here comes the augmentation. for the demo translate the defect in the center of separator mask - deprecated
        # cv2.imwrite("target_separator_mask.png", target_separator_mask)
               
        target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = (400,400))
        # target_separator_mask = erode_mask_single(target_separator_mask, kernel_dims = (200,200))
        # cv2.imwrite("target_separator_mask_shrinked.png", target_separator_mask)
  
        mesh, img_draw = mask_to_mesh(target_separator_mask, target_img)
        # cv2.imwrite("img_draw.png", img_draw)

        #choose a random x,y pair from mesh to translate defect
        X, Y = mesh
        positions = np.vstack([X.ravel(), Y.ravel()])
        rand_i = np.random.choice(positions.shape[1], replace=False)
        tx = positions[0][rand_i]
        ty = positions[1][rand_i]       
        # (tx,ty),_ = find_centroid_of_mask(target_separator_mask)
        
        source_defect_mask_to_top_left_rs_adj_translated = translate_image(source_defect_mask_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        # cv2.imwrite("source_hole_mask_to_top_left_rs_adj_translated.png", source_hole_mask_to_top_left_rs_adj_translated)
        
        source_defect_img_to_top_left_rs_adj_translated = translate_image(source_defect_img_to_top_left_rs_adj, (tx-tx_), (ty - ty_))
        # cv2.imwrite("source_hole_img_to_top_left_rs_adj_translated.png", source_hole_img_to_top_left_rs_adj_translated)
    
        (fx, fy), _ = find_centroid_of_mask(source_defect_mask_to_top_left_rs_adj_translated)
        center = (int(fx), int(fy))

        angles = np.arange(0,360)
        rand_angle = random.choice(angles)
        rot_source_defect_mask_to_top_left_rs_adj_translated =  rotate_mask(source_defect_mask_to_top_left_rs_adj_translated, center, rand_angle)
        rot_source_defect_img_to_top_left_rs_adj_translated =  rotate_mask(source_defect_img_to_top_left_rs_adj_translated, center, rand_angle)

        alpha = 0.85
        mask_ = (rot_source_defect_mask_to_top_left_rs_adj_translated/255).astype(float)
        mask__ = 1 - alpha*mask_
        target_img_ = cv2.multiply(target_img.astype(float), mask__)
        target_img_ = cv2.add(target_img_, rot_source_defect_img_to_top_left_rs_adj_translated.astype(float))
        new_path = P_2(defect_dir, f"{int(i/2)}_defect.jpg")
        cv2.imwrite(new_path, target_img_.astype(np.uint8))

        list_of_lists_points = mask_to_shape(rot_source_defect_mask_to_top_left_rs_adj_translated[:, :, 0])

        if type_of_defect == "hole":
            label = "hole"
        elif type_of_defect == "tearing":
            label = "tearing"

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
        with open(f"{defect_dir}/{int(i/2)}_defect.json", 'w') as f:
            # Write the modified data back to the file
            json.dump(data, f)


        break
    return 1


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

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
        
    # s_center,dims = find_centroid_of_mask(mask.astype(bool))
    # t_center = (mask.shape[1]//2, mask.shape[0]//2)
    # if len(mask.shape)==3:
    #     mask_3channels = mask
    # else:
    #     mask_3channels = cv2.merge((mask,mask,mask))
    # tx = t_center[0] - s_center[0]
    # ty = t_center[1] - s_center[1]
    # mask_translated = translate_image(mask_3channels, tx, ty)
    # if len(mask.shape)==3:
    #     return mask_translated[:,:,:]
    # else:
    #     return mask_translated[:,:,0]

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
 


def get_random_around_centers(contour_info, num_centers, in_bbox_region=False):
    #np.random.seed(11)
    contour_x0,contour_y0,contour_xt,contour_yt,contour_center,contour_mask = (contour_info['x0'], contour_info['y0'], contour_info['xt'], contour_info['yt'], contour_info['center'], contour_info['mask'])
    if in_bbox_region:#search random points inside bbbox of contour(might find a point outside of contour)
        contour_center_x = contour_center[0]
        contour_center_y = contour_center[1]
        x_pixel_choices = np.random.choice(np.arange(contour_x0, contour_xt, 1), num_centers)
        y_pixel_choices = np.random.choice(np.arange(contour_y0, contour_yt, 1), num_centers)
    else:
        x_pixel_choices = np.random.choice(np.where(contour_mask)[1], num_centers)
        y_pixel_choices = np.random.choice(np.where(contour_mask)[0], num_centers)
    centers = [(i,j) for i,j in zip(x_pixel_choices, y_pixel_choices)]   
    # print('centers: {}'.format(centers))
    if num_centers==1:
        centers = centers[0]
    return centers
    # w,h = window_size
    # image_xmin,image_ymin,image_xmax,image_ymax = limit_xy
    #calcluate limits for the augmented centers to not exceed image limits    
    # x0 = contour_x0 - w/2#this is the minimum x of bbox of contour, around the contour points
    # xt = contour_xt + w/2
    # y0 = contour_y0 - h/2
    # yt = contour_yt + h/2   
    #check if window_size is applicable
 
    # for center in centers:
    #     xc_0 = center[0] 
    #     yc_0 = center[1]
        
    # # if x0<image_xmin:
    # #     augm_center_xmin = contour_x0 +image_xmin - x0
    # # if xt>image_xmax:
    # #     augm_center_xmax = contour_xt - (xt- image_xmax)
    # # if y0<image_ymin:
    # #     augm_center_ymin = contour_y0 + image_ymin - y0
    # # if yt>image_ymax:
    # #     augm_center_ymax = contour_yt - (yt - image_ymax)

    # print('random augmented x centers will be chosen from:{}-{}, random augmented y centers will be chosen from: {}-{}'.format(augm_center_xmin,augm_center_xmax,augm_center_ymin,augm_center_ymax))
    # x_pixel_choices = np.random.choice(np.arange(augm_center_xmin, augm_center_xmax, 1), num_centers)
    # y_pixel_choices = np.random.choice(np.arange(augm_center_ymin, augm_center_ymax, 1), num_centers)
    # centers = [(i,j) for i,j in zip(x_pixel_choices, y_pixel_choices)]       


def get_rectangle_max_info(rectangle):
    #rectangle is cv format: (topleft, bottomright)
    xmin = rectangle[0]
    ymin = rectangle[1]
    xmax = rectangle[2]
    ymax = rectangle[3]
    return (xmin,xmax,ymin,ymax)
    
def rectangles_from_centers(centers_list_tuples, window):
    #rectangle is cv format: (topleft, bottomright)
    #centers_tuple = (x,y)
    rectangles = []
    w = window[0]
    h = window[1]
    for center in centers_list_tuples:
        top_left_x = int(center[0] - w/2)
        top_left_y = int(center[1] - h/2)
        bottom_right_x = int(center[0] + w/2)
        bottom_right_y = int(center[1] + h/2)
        rectangles.append([(top_left_x, top_left_y),(bottom_right_x, bottom_right_y)])
    return rectangles

def rectangles_from_contour_slide(image_dims, contour,window_size,overlap):
    rectangles = []
    idx = 0
    x0,y0,w,h = cv2.boundingRect(contour)
    #x0,y0 is top left corner
    window_x = window_size[0]
    window_y = window_size[1]
    
    if window_x>w or window_y>h:
        print("error")
    step_x = int((1-overlap)*window_x)
    step_y = int((1-overlap)*window_y)
    moves_x_1 = np.arange(x0, x0 + w, step_x)
    moves_y_1 = np.arange(y0, y0 + h, step_y)
    for move_x_1 in moves_x_1:
        for move_y_1 in moves_y_1:
            if (move_y_1 + window_y )> image_dims[0]:
                move_y_1 = image_dims[0] - window_y 
            if (move_x_1 + window_x )> image_dims[1]:
                move_x_1 = image_dims[1] - window_x 
            top_left_x = int(move_x_1)
            top_left_y = int(move_y_1)
            bottom_right_x = int(move_x_1 + window_x)
            bottom_right_y = int(move_y_1 + window_y)
            rectangles.append([(top_left_x, top_left_y),(bottom_right_x, bottom_right_y)])
    return rectangles

def count_defects_from_cropped_rectangles(rectangles, image_data, masks_dict,labels, window_size):
    window_x = window_size[0]
    window_y = window_size[1]
    defect_no = 0
    nondefect_no = 0
    for rect in rectangles:
        img_copy = copy.deepcopy(image_data)  
        crop = image_data[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] #toplefty:bottomrighty, topleftx, bottomrightx           
        if crop.shape[0]==0 or crop.shape[1]==0:
            print('if print this, needs debugging..')
            continue
        crop_info = (rect[0][0], rect[0][1], window_x, window_y)
        new_mask_dict = crop_mask_dict(masks_dict, crop_info)
        
        temp = masks_dict_to_labelme_dict(new_mask_dict, crop.shape[:2],labels, 'temp not used')
        json_has_defect = has_dict_these_categories(temp,'defect')
        if json_has_defect:
            defect_no+=1
        else:
            nondefect_no+=1
    return defect_no, nondefect_no
            
    

def rectangles_from_centers_fit_inside_limits(contour_info, window, limit_xy, num_centers):
    #rectangle is cv format: (topleft, bottomright)
    #centers_tuple = (x,y)
    rectangles = []
    w = window[0]
    h = window[1]
    image_xmin,image_ymin,image_xmax,image_ymax = limit_xy
    count_centers = 0
    contour_x0,contour_y0,contour_xt,contour_yt,contour_center,contour_mask = (contour_info['x0'], contour_info['y0'], contour_info['xt'], contour_info['yt'], contour_info['center'], contour_info['mask'])
    while count_centers<num_centers:
        rand_center = get_random_around_centers(contour_info, num_centers = 1, in_bbox_region = False)
        # print('rand_center: {}'.format(rand_center))
        top_left_x = int(rand_center[0] - w/2)
        top_left_y = int(rand_center[1] - h/2)
        bottom_right_x = int(rand_center[0] + w/2)
        bottom_right_y = int(rand_center[1] + h/2)
        out_of_xmin = 0#how many pixels outside of image limits is the window crop with the random center inside the contour
        out_of_xmax = 0
        out_of_ymin = 0
        out_of_ymax = 0
        if top_left_x<image_xmin:
            out_of_xmin = image_xmin - top_left_x
        if bottom_right_x>image_xmax:
            out_of_xmax = bottom_right_x - image_xmax
        if top_left_y<image_ymin:
            out_of_ymin = image_ymin - top_left_y
        if bottom_right_y>image_ymax:
            out_of_ymax = bottom_right_y - image_ymax 
            
        if (out_of_xmin>0 and out_of_xmax>0):
            print("not applicable, check window size not exceed image limits")
        if (out_of_ymin>0 and out_of_ymax>0):
            print("not applicable, check window size not exceed image limits")
            
        # print('top_left_x: {}, bottom_right_x:{}, top_left_y:{}, bottom_right_y:{}'.format(top_left_x,bottom_right_x,top_left_y,bottom_right_y))     
        top_left_x = top_left_x + out_of_xmin  - out_of_xmax
        bottom_right_x = bottom_right_x - out_of_xmax + out_of_xmin
        top_left_y = top_left_y + out_of_ymin- out_of_ymax 
        bottom_right_y = bottom_right_y - out_of_ymax  + out_of_ymin
        # print('top_left_x: {}, bottom_right_x:{}, top_left_y:{}, bottom_right_y:{}'.format(top_left_x,bottom_right_x,top_left_y,bottom_right_y))
        # print('new center:{},{}'.format(top_left_x +w/2, top_left_y+h/2))
        rectangles.append([(top_left_x, top_left_y),(bottom_right_x, bottom_right_y)])
        count_centers+=1
    return rectangles


def crop_same_images_scatter(scanfolder, savefolder, imgfiles, limit_region_mask, window_size, camera_and_sink_id = 'None', debug_vis = True, num_augmented_center_crops = 10):
    #img files are images of the same sink, same camera but different illumination
    #window_size = (w,h)
    #get dims, sink mask and defect mask from one random images from imgfiles since they are all the same but with differnt illumination
    image_data = cv2.imread(P_2(scanfolder, imgfiles[0]))
    image_limits = (0,0,image_data.shape[1],image_data.shape[0])
    json_i = img2json(imgfiles[0])
    dims = image_data.shape[:2]
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(0,20)]
    add_crop_idx_to_img_name = lambda x,y:x.split(".")[0]+"_"+y+".jpg"
    add_label_info_to_img_name = lambda x,y:x.split(".")[0]+"_"+y+".jpg"
    window_x = window_size[0]
    window_y = window_size[1]    
    crops = {}
    masks_dict, labels= json_to_masks(scanfolder, json_i, dims, booltype=False)
    if 'defect' not in labels:
        return    
    try:
        sink_mask = masks_dict['sink'][0]
    except KeyError:
        pass
    sink_contour = mask_to_contours(sink_mask)[0]
    # sink_limits = get_contour_info(sink_contour)
    # sink_centroid = find_centroid_of_mask(sink_mask)
    # sx = sink_centroid[0]
    # sy = sink_centroid[1]
    # #make a contour using sink centroid and window_size
    # tl_x = int(sx - window_x/2)
    # tl_y = int(sy - window_y/2)
    # tr_x = int(tl_x + window_x)
    # tr_y = int(tl_y)
    # br_x = int(tr_x)
    # br_y = int(tl_y + window_y)
    # bl_x = int(tl_x)
    # bl_y = int(br_y)
    # contour_window = [[[tl_x, tl_y]],[[tr_x,tr_y]],[[br_x,br_y]],[[bl_x,bl_y]]]
    # sink_center_rect = get_contour_info(contour_window)
    # img2 = cv2.rectangle(copy.deepcopy(image_data),(sink_center_rect[0], sink_center_rect[1]), (sink_center_rect[2], sink_center_rect[3]),colors[np.random.choice(len(colors))],9)
    # cv2.imwrite("tttt.jpg", img2)
    
    defect_masks = masks_dict['defect']
    defect_contours = [mask_to_contours(i)[0] for i in defect_masks]
    total_augment_centers = []
    total_augment_rectangles = []
    for contour in defect_contours:
        center = find_centroid_contour(contour)
        # percentage_beyond_bbox_of_contour=0.0

        #GET 3 AUGMENTATIONS: 1. AROUND THE DEFECT, 2. AROUND THE DEFECT LOOSE 3. RAND IN IMAGE        
        contour_info = get_contour_info(contour,image_data.shape[:2], expand = 0.0)
        scattered_rectangles_augment = rectangles_from_centers_fit_inside_limits(contour_info, window_size, image_limits, num_augmented_center_crops)
        total_augment_rectangles.extend(scattered_rectangles_augment)
        
        
        # augment_centers = get_random_around_centers(contour_info, num_augmented_center_crops)   
        # total_augment_centers.extend(augment_centers)

        # percentage_beyond_bbox_of_contour=2.0
        # contour_info = get_contour_info(contour, expand = percentage_beyond_bbox_of_contour)
        # augment_centers = get_random_around_centers(contour_info, num_augmented_center_crops)   
        # total_augment_centers.extend(augment_centers)
        
        # percentage_beyond_bbox_of_contour=0.0#3. get random points in sink_contour region
        # contour_info = get_contour_info(sink_contour, expand = percentage_beyond_bbox_of_contour)
        # augment_centers = get_random_around_centers(contour_info, num_augmented_center_crops)   
        # total_augment_centers.extend(augment_centers)
        
    # augment_rectangles= rectangles_from_centers(total_augment_centers, window_size,num_augmented_center_crops)
    #apply same augment_rectangles to all images in imgfiles

    for i in tqdm(range(0,len(imgfiles))):#imgfiles may contain same images from differetn illumination levels
        imgfile = imgfiles[i]
        image_data = cv2.imread(P_2(scanfolder, imgfile))
        idx=0
        for rect in total_augment_rectangles:
            img_copy = copy.deepcopy(image_data)  
            crop = image_data[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] #toplefty:bottomrighty, topleftx, bottomrightx           
            if crop.shape[0]==0 or crop.shape[1]==0:
                print('if print this, needs debugging..')
                continue
            crop_info = (rect[0][0], rect[0][1], window_x, window_y)
            new_mask_dict = crop_mask_dict(masks_dict, crop_info)
            
            temp = masks_dict_to_labelme_dict(new_mask_dict, crop.shape[:2],labels, 'temp not used')
            json_has_defect = has_dict_these_categories(temp,'defect')
            if json_has_defect:
                label_info = "DEF"
            else:
                label_info = "NODEF"    
            print('label_info: {}'.format(label_info))
            imgfile_crop_name = add_crop_idx_to_img_name(imgfile, str(idx))
            imgfile_crop_name = add_label_info_to_img_name(imgfile_crop_name, label_info)
            jsonfile_crop = img2json(imgfile_crop_name)
            new_mask_dict_json = masks_dict_to_labelme_dict(new_mask_dict, crop.shape[:2],labels, jsonfile_crop)
            with open(P_2(savefolder, jsonfile_crop),'w') as f:
                json.dump(new_mask_dict_json, f, indent=2)  
            cv2.imwrite(P_2(savefolder,imgfile_crop_name), crop)
            if debug_vis:
                img2 = cv2.rectangle(copy.deepcopy(img_copy),rect[0], rect[1],colors[np.random.choice(len(colors))],9)
                show(img2,4)                                                                           
            crops[str(idx)] = [rect[0][0], rect[1][0], rect[0][1], rect[1][1]]
            idx+=1
    save_dict_to_json(savefolder, "{}_reconstruct_coords.json".format(camera_and_sink_id), crops)
   
    # contour = mask_to_contours(limit_region_mask)
    # x0,y0,w,h = cv2.boundingRect(contour[0])
    # #x0,y0 is top left corner
    # window_x = window_size[0]
    # window_y = window_size[1]
    return crops


        
