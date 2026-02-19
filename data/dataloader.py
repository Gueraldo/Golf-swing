import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

class SwingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mask_shape, transform=None, im_transform=None):
        with open(annotations_file, 'r') as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.img_dir = img_dir
        self.transform = transform
        self.im_transform = im_transform
        self.mask_shape = mask_shape
        self.sigma = 2
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

        self.samples = []
        for img_idx, img in enumerate(self.images):
            anns = self.annotations_by_img.get(img["id"], [])

            for ann_idx in range(len(anns)):
                self.samples.append((img_idx, ann_idx))

        print("Total person samples:", len(self.samples))

    def __len__(self):
        return len(self.images)
    
    def generate_heatmaps(self, keypoints, orig_size):
        ###
        # Places a heat signature on the heatmap
        # at the location x, y of the keypoint
        ###
        num_keypoint = len(self.labels_name)
        heatmaps = np.zeros((num_keypoint,
                             self.mask_shape[0],
                             self.mask_shape[1]),
                            dtype=np.float32)
        
        scale_x = self.mask_shape[1] / orig_size[1]
        scale_y = self.mask_shape[0] / orig_size[0]

        for j in range(num_keypoint):
            x, y, v = keypoints[j*3:j*3+3]
            if v == 0:
                continue

            x *= scale_x
            y *= scale_y

            self.draw_gaussian(heatmaps[j], x, y)

        return heatmaps
    
    def draw_gaussian(self, heatmap, x, y):
        tmp_size = self.sigma * 3

        mu_x = int(x + 0.5)
        mu_y = int(y + 0.5)

        w, h = heatmap.shape[1], heatmap.shape[0]

        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
            return

        size = 2 * tmp_size + 1
        x_coords = np.arange(0, size, 1, np.float32)
        y_coords = x_coords[:, None]
        g = np.exp(-( (x_coords - tmp_size)**2 +
                    (y_coords - tmp_size)**2 ) / (2 * self.sigma**2))

        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]

        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        

    def __getitem__(self, idx):
        # get person sample
        img_idx, ann_idx = self.samples[idx]

        img_info = self.images[img_idx]
        anns = self.annotations_by_img[img_info["id"]]
        ann = anns[ann_idx]

        # load image
        image = cv2.cvtColor(
            cv2.imread(img_info["file_name"]),
            cv2.COLOR_BGR2RGB
        )

        orig_h, orig_w = image.shape[:2]

        # PERSON CROP USING BBOX
        x, y, w, h = ann["bbox"]

        # expand bbox
        scale = 1.25
        cx = x + w / 2
        cy = y + h / 2

        new_w = w * scale
        new_h = h * scale

        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x2 = int(cx + new_w / 2)
        y2 = int(cy + new_h / 2)

        # handle borders
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)
        pad_x2 = max(0, x2 - orig_w)
        pad_y2 = max(0, y2 - orig_h)

        image = cv2.copyMakeBorder(
            image,
            pad_y1, pad_y2,
            pad_x1, pad_x2,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        x1 += pad_x1
        x2 += pad_x1
        y1 += pad_y1
        y2 += pad_y1

        cropped = image[y1:y2, x1:x2]

        # TRANSFORM KEYPOINTS TO CROPPED IMAGE COORDS

        keypoints = ann["keypoints"].copy()

        for j in range(len(self.labels_name)):
            keypoints[j*3]   -= x1  # x
            keypoints[j*3+1] -= y1  # y

        crop_h, crop_w = cropped.shape[:2]

        # GENERATE HEATMAPS FROM CROPPED PERSON

        masks = self.generate_heatmaps(keypoints, (crop_h, crop_w))

        # OPTIONAL AUGMENTATIONS

        if self.transform is not None:
            transformed = self.transform(image=cropped, masks=masks)
            image = transformed["image"]
            masks = transformed["masks"]
        else:
            image = cropped

        if self.im_transform is not None:
            im_transformed = self.im_transform(image=image)
            image = im_transformed["image"]

        image = np.array(image, dtype=np.float32)

        return image, masks