import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter
import albumentations as A
import random

def albumentationAugmenter(image,mask,epochs):
    crop=random.choice(list(range(256,1024,2)))
    # ReplayCompose, so that we can record which augmentations are used
    transform = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.25, brightness_limit=0.2, contrast_limit=0.2),
        A.Blur(blur_limit=5, p=0.25),
        A.GaussNoise(p=0.25,var_limit=(0.001, 0.001))
    ])
    transformed=transform(image=image, mask=mask)
    return(transformed['image'], transformed['mask'], transformed['replay'],1,crop)
