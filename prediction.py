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
#import model1
import simulationV2
import numpy as np
import cv2
import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import Sampler



def predict(model, imageDir, device):

    print("DO I PREDICT!?")

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    pred_set = GetData("predict", imageDir, transform=trans)
    
    batch_size = 2
    
    test_loader = DataLoader(pred_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()

    files = os.listdir(imageDir)


    for i in range(0,len(pred)):
        newFile = files[i].replace(".png","_mask.png")
        newFile = files[i].replace(".jpg","_mask.png")

        newPred = pred[i][0]
    
        newPred[newPred > 0.5] = 255
        newPred[newPred <= 0.5] = 0
        plt.imsave(imageDir+newFile, newPred)

    return pred

