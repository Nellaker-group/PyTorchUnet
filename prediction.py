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

from Dataset import GetDataMontage, GetDataFolder, GetDataSeqTilesFolderPred
from polygoner import draw_polygons_from_mask

# this function assumes that the tiling was done with tileWSI.sh
def addMaskgetXY(filename):
    assert "/" not in filename
    filename=filename.replace(".png","_mask.png").replace(".jpg","_mask.png")
    X = int(filename.split("_X")[1].split("_Y")[0])
    Y = int(filename.split("_Y")[1].split("_mask")[0])
    return(filename, X, Y)

def predict(model, preDir, imageDir, device, preName, normFile, inputChannels, zoomFile, whichDataset, predThreshold, predMaskDir):

    print("DO I PREDICT!?")

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    sample_size_pred = len(os.listdir(preDir))
    pred_set = GetDataSeqTilesFolderPred("predict", preName, normFile, inputChannels, zoomFile, whichDataset, pathDir=preDir, transform=trans)
    batch_size = 2
    test_loader = DataLoader(pred_set, batch_size=batch_size, shuffle=False, num_workers=0)    
    filesWritten=0

    files = os.listdir(preDir)
    files = [s for s in files if not "_mask.png" in s]
    files.sort()

    polygonList = []

    for inputs, labels, filenames in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Predict
        pred = model(inputs)
        # The loss functions include the sigmoid function.
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        
        # first element of the batch
        newFilename, X, Y = addMaskgetXY(filenames[0])
        newPred = pred[0][0]
        newPred[newPred > predThreshold] = 255
        newPred[newPred <= predThreshold] = 0
        # put masks in new directory
        # and geojson file
        plt.imsave(predMaskDir+newFilename, newPred)
        filesWritten+=1
        polygons, segmentations = draw_polygons_from_mask(newPred,X,Y)
        polygonList.append(polygons)

        # check that there files left to be predicted on, since we have batch of two
        if((filesWritten+1)>len(files)):
           break

        # second element of the batch
        newFilename, X, Y = addMaskgetXY(filenames[1])
        newPred = pred[1][0]
        newPred[newPred > predThreshold] = 255
        newPred[newPred <= predThreshold] = 0
        plt.imsave(predMaskDir+newFilename, newPred)
        filesWritten+=1
        polygons, segmentations = draw_polygons_from_mask(newPred,X,Y)
        polygonList.append(polygons)

    polygonList = [x for xs in polygonList for x in xs]
    return polygonList


