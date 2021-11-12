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

from Dataset import GetDataTilesArray, GetDataSeqTilesFolder
from Sampler import RandomSampler, SeqSamplerDatasetSize, SeqSamplerUniform

def predict(model, pathDir, imageDir, device, preName):

    print("DO I PREDICT!?")

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    sample_size_pred = len(os.listdir(pathDir))

    pred_set = GetDataSeqTilesFolder("predict", preName, pathDir=pathDir, transform=trans)
    
    batch_size = 2

    assert batch_size == 2

    test_loader = DataLoader(pred_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    filesWritten=0

    files = os.listdir(pathDir)
    files = [s for s in files if not "_mask.png" in s]
    files.sort()

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Predict
        pred = model(inputs)
        # The loss functions include the sigmoid function.
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()

        # first element of the batch
        newFile = files[filesWritten].replace(".png","_mask.png")
        newFile = newFile.replace(".jpg","_mask.png")
        newPred = pred[0][0]
        # Emil - which one to use!?
        newPred[newPred > 0.5] = 255
        newPred[newPred <= 0.5] = 0
        # newPred[newPred > 0.9] = 255
        # newPred[newPred <= 0.9] = 0
        plt.imsave(pathDir+newFile, newPred)
        filesWritten+=1

        # second element of the batch
        newFile = files[filesWritten].replace(".png","_mask.png")
        newFile = newFile.replace(".jpg","_mask.png")
        newPred = pred[1][0]
        # Emil - which one to use!?
        newPred[newPred > 0.5] = 255
        newPred[newPred <= 0.5] = 0
        # newPred[newPred > 0.9] = 255
        # newPred[newPred <= 0.9] = 0
        plt.imsave(pathDir+newFile, newPred)
        filesWritten+=1

    return pred

