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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def dilate(in_channels, out_channels, dilation):
    return nn.Conv2d(in_channels, out_channels, dilation)     

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()                
        self.dconv_down1 = double_conv(1, 44)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_down2 = double_conv(44, 44*2)
        self.dconv_down3 = double_conv(44*2, 44*4)       
        self.dilate1 = nn.Conv2d(44*4, 44*8, 3, dilation=1)     
        self.dilate2 = nn.Conv2d(44*8, 44*8, 3, dilation=2)     
        self.dilate3 = nn.Conv2d(44*8, 44*8, 3, dilation=4)     
        self.dilate4 = nn.Conv2d(44*8, 44*8, 3, dilation=8)     
        self.dilate5 = nn.Conv2d(44*8, 44*8, 3, dilation=16)     
        self.dilate6 = nn.Conv2d(44*8, 44*8, 3, dilation=32)     
        self.upsample = nn.Upsample(scale_factor=2)        
        self.dconv_up3 = double_conv(44*8, 44*4)
        self.dconv_up2 = double_conv(44*4+44*2, 44*2)
        self.dconv_up1 = double_conv(44*2+44, 44)        
        self.conv_last = nn.Conv2d(44, n_class, 1)
        
        
    def forward(self, x):        
        # to convert the Tensor to have the data type of floats
        x = x.float()
        #x = torch.reshape(x, (3, 1024, 1024))
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        x1 = self.dilate1(x)
        x2 = self.dilate2(x1)
        x3 = self.dilate3(x2)
        x4 = self.dilate4(x3)
        x5 = self.dilate5(x4)
        x6 = self.dilate6(x5)
        # I AM NOT SURE HOW TO DO THIS
        # ASK ABOUT https://github.com/GlastonburyC/Adipocyte-U-net/blob/master/src/models/adipocyte_unet.py
        # Line 138-145
        # x = add([x1, x2, x3, x4, x5, x6])        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)           
        x = self.dconv_up1(x)        
        out = self.conv_last(x)        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
