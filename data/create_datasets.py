# This file creates the heatmaps datasets in which there are as many folders as joints in the pose dataset.

import numpy as np
import cv2
import os
from dataloader import SwingDatasetCreation

def placeGaussian(input_map, x, y):
    ###
    # Plances a heat signature on the unput heatmap
    # at the location x, y
    ###
    sigma = 5
    map_shape = np.shape(input_map)
    
    xx, yy = np.meshgrid(np.arange(map_shape[1]), np.arange(map_shape[0]))
    gauss = 1 / (np.pi * 2 * sigma ** 2) * np.exp(- (np.square(xx - x) + np.square(yy - y)) / (2 * sigma ** 2))
    gauss *= 255

    np.maximum(input_map, gauss, out=input_map)


def main():
    labels_name = [
        "r_ankle",
        "r_klnee",
        "r_hip",
        "l_hip",
        "l_knee", 
        "l_ankle",
        "pelvis", 
        "thorax", 
        "upper_neck", 
        "head_top", 
        "r_wrist",
        "r_elbow",
        "r_shoulder",
        "l_shoulder",
        "l_elbow",
        "l_wrist"
    ]

    dataset = SwingDatasetCreation("annotations/train.json", "data")
    print(f"len of dataset = {len(dataset)}")
    mask_shape = (134, 174)
    kp_masks = np.zeros((16, mask_shape[0], mask_shape[1]))
    
    for name in labels_name:
        try:
            directory_name = "annotations/" + name
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return
    
    for idx in range(len(dataset)):
        label = dataset.annotations_by_img.get(dataset.images[idx]["id"], [])
        kp_masks = np.zeros((16, mask_shape[0], mask_shape[1]))
        im_name = dataset.images[idx]["file_name"]
        _, im_name = os.path.split(im_name)

        for ann in label:
            
            scale = [mask_shape[1] / dataset.images[idx]["width"], mask_shape[0] / dataset.images[idx]["height"]]

            for j in range(16):
                if ann["keypoints"][j * 3 + 2]: # if the keypoint is visible
                    x, y = ann["keypoints"][j * 3 + 0] * scale[0], ann["keypoints"][j * 3 + 1] * scale[1]
                    placeGaussian(kp_masks[j], x, y)
        
        for i in range(16):
            # Normalize heatmap to 0-255 and convert to uint8
            heatmap = kp_masks[i]
            heatmap_norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
            heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

            save_path = os.path.join("annotations", labels_name[i], im_name)
            try:
                success = cv2.imwrite(save_path, heatmap_uint8)
                if not success:
                    print(f"Failed to save heatmap for {labels_name[i]} at {save_path}")
            except Exception as e:
                print(f"Error saving heatmap for {labels_name[i]} at {save_path}: {e}")

        
    

if __name__ == "__main__":
    main()