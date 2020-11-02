import os
import glob
import torch

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from collections import defaultdict
from torchvision import transforms

from albumentations import (
    Compose, 
    OneOf,
    RandomBrightnessContrast,
    RandomGamma, 
    ShiftScaleRotate
)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIM_ACRDataset(torch.utils.data.Dataset):
    def __init__(self,
        image_ids,
        arguments,
        transform=True,
        preprocessing_fn=None):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the image, list
        :param transform: True/False, no transform in validation
        :param preprocessing_fn: a function for preprocessing image
        """
        # we create a empty dictionary 
        # and mask paths
        self.data = defaultdict(dict)
        self.counter = 0
        
        # get the arguments
        self.args = arguments

        # for augmentation
        self.transform = transform

        # preprocessing function 
        self.preprocessing_fn = preprocessing_fn

        # albumentation augmentation
        # we have shift, scale & rotate
        # applied with 80% probability
        # and then one of gamma and brightness/contrast
        # is applied to the image
        # albumentation takes care of which augmentation
        # is applied to image and mask
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10, p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )
        # going pver all image_ids to store 
        # image and mask paths
        for imgid in image_ids:
            imgid = imgid.split('.')[0] 
            self.data[self.counter] = {
                "img_path": os.path.join(
                    self.args.image_path, imgid + ".png"
                ),
                "mask_path": os.path.join(
                    self.args.mask_path, imgid + ".png"
                ),
            }
            self.counter+=1
        
    def __len__(self):
        # return length of dataset
        return len(self.data)
    
    def __getitem__(self, item):
        # for a givenm item index,
        # return image and mask tensors
        # read image and mask paths
        img_path = self.data[item]["img_path"]
        #print("img_path", img_path)
        mask_path = self.data[item]["mask_path"]

        # read image and convert to RGB
        img = Image.open(img_path)
        img = img.resize((self.args.resize_img, self.args.resize_img))
        img = img.convert("RGB")

        # PIL image to numpy array
        img = np.array(img)

        # read mask image
        mask = Image.open(mask_path)
        mask= mask.resize((self.args.resize_img, self.args.resize_img))
        mask = np.array(mask)

        # Convert to binary float matrix
        mask = (mask>=1).astype("float32")

        # if this is training data, apply transforms
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # preprocess the image using provided 
        # preprocessing tensors tis is manually
        # image normalization
        img = self.preprocessing_fn(img)

        # return image and mask tensors
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float(),
        } 

