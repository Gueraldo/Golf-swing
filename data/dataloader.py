import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

class SwingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, im_transform=None):
        with open(annotations_file, 'r') as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.img_dir = img_dir
        self.transform = transform
        self.im_transform = im_transform
        self.mask_shape = (134, 174)
        self.labels_name = [
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
        
        self.annotations_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_img[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]["file_name"]), cv2.COLOR_BGR2RGB)

        kp_masks = np.array([
            cv2.resize(
                cv2.imread(
                    os.path.join("data", "annotations", self.labels_name[i], os.path.split(self.images[idx]["file_name"])[1]),
                    cv2.IMREAD_GRAYSCALE
                ),
                np.flip(np.array(self.mask_shape))
            ) for i in range(16)
        ], dtype=np.float32)

        if self.transform is not None:
            transformed  = self.transform(image=image, masks=kp_masks)
            image = transformed["image"]
            masks = transformed["masks"]

        if self.im_transform is not None:
            im_transformed = self.im_transform(image=image)
            image = im_transformed["image"]

        # Resizing and normalizing the masks
        
        masks = np.array(masks) / 255.0
        
        

        # super_imposed_img = image
        # for i in range(16):
        #     # Ensure mask is in uint8 and has 3 channels
        #     mask = masks[i]
        #     if mask.dtype != np.uint8:
        #         mask = (mask / mask.max() * 255).astype(np.uint8) if mask.max() > 0 else np.zeros_like(mask, dtype=np.uint8)
            
        #     color_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        #     color_map = cv2.resize(color_map, (640, 480))
            
        #     # Resize image to match mask size and ensure it is also uint8
        #     base_img = cv2.resize(super_imposed_img, (640, 480))
        #     if base_img.dtype != np.uint8:
        #         base_img = (base_img * 255).astype(np.uint8) if base_img.max() <= 1.0 else base_img.astype(np.uint8)

        #     # Blend the two
        #     super_imposed_img = cv2.addWeighted(color_map, 0.05, base_img, 0.95, 0)

        # plt.figure()
        # plt.imshow(super_imposed_img)
        # plt.show()
        # cv2.imshow("super_imposed_img", super_imposed_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        image = np.array(image, dtype=np.float32)

        return image, masks
    

class SwingDatasetCreation(Dataset):
    def __init__(self, annotations_file, img_dir):
        with open(annotations_file, 'r') as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.img_dir = img_dir
        
        self.annotations_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_img[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_shape = (self.images[idx]["height"], self.images[idx]["width"])
        label = self.annotations_by_img.get(self.images[idx]["id"], [])

        return im_shape, label
    

def placeGaussian(input_map, x, y):
    ###
    # Plances a heat signature on the unput heatmap
    # at the location x, y
    ###
    sigma = 1.5
    map_shape = np.shape(input_map)
    
    xx, yy = np.meshgrid(np.arange(map_shape[1]), np.arange(map_shape[0]))
    gauss = 1 / (np.pi * 2 * sigma ** 2) * np.exp(- (np.square(xx - x) + np.square(yy - y)) / (2 * sigma ** 2))

    np.maximum(input_map, gauss, out=input_map)