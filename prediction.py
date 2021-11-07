from collections import defaultdict
from loss import dice_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
import cv2
import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import Sampler

from Dataset import GetDataSeqTilesArray, GetDataTilesArray, GetDataSeqTilesFolder
from Sampler import RandomSampler, SeqSampler

def predict(model, pathDir, imageDir, device):

    print("DO I PREDICT!?")

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    sample_size_pred = len(os.listdir(pathDir))

    if imageDir:
        pred_set = GetDataSeqTilesFolder("predict", pathDir, transform=trans)
    else:
        pred_set = GetDataTilesArray("predict", pathDir, transform=trans)
    
    batch_size = 2

    samplie_pred = RandomSampler(pred_set, sample_size_pred, 1024, 0)

    if imageDir:
        test_loader = DataLoader(pred_set, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        test_loader = DataLoader(pred_set, batch_size=batch_size, num_workers=0, sampler=samplie_pred)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()

    files = os.listdir(pathDir)


    for i in range(0,len(pred)):
        newFile = files[i].replace(".png","_mask.png")
        newFile = newFile.replace(".jpg","_mask.png")
        newPred = pred[i][0]

        # Emil - which one to use!?
        newPred[newPred > 0.5] = 255
        newPred[newPred <= 0.5] = 0
        # newPred[newPred > 0.9] = 255
        # newPred[newPred <= 0.9] = 0

        plt.imsave(pathDir+newFile, newPred)

    return pred

