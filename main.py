from collections import defaultdict
#from loss import dice_loss
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

import loss
import Sampler
import Dataset
import prediction
import training
import DataLoader
import model

from model import UNet
from model import count_parameters
from DataLoader import get_dataloader
from training import train_model
from prediction import predict

def select_gpu(whichGPU):

    if(whichGPU=="0"):
        print("here0")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif(whichGPU=="1"):
        print("here1")
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif(whichGPU=="2"):
        print("here2")
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    elif(whichGPU=="3"):
        print("here3")
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        
    return device

def main():

    ## index, 0 will give you the python filename being executed. Any index after that are the arguments passed.
    gpu= sys.argv[1]
    trainOrPredict= sys.argv[2]
    seed=int(sys.argv[3])
    np.random.seed(seed)

    if(len(sys.argv)>3):
        imageDir= sys.argv[4]

    print("IT IS ASSUMED THAT THIS SCRIPT IS EXECUTED FROM THE DIRECTORY OF THE FILE")

    assert os.path.isfile("main.py") and os.path.isfile("DataLoader.py") and os.path.isfile("loss.py") 

    assert trainOrPredict in ['train', 'predict']

    device = select_gpu(gpu)
    print(device)
        
    num_class = 1
    model = UNet(n_class=num_class).to(device)
        
    #model.double()
    print(model)
    
    print("N parameters:")
    print(count_parameters(model))

    if(trainOrPredict == "train"):
        training_data = get_dataloader()
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
        model = train_model(model, training_data, device, optimizer_ft, exp_lr_scheduler, num_epochs=60)
        if os.path.isdir('weights/'): 
            torch.save(model.state_dict(),"weights/weightsRandomTiles.dat")
        else:
            os.mkdir('weights/') 
            torch.save(model.state_dict(),"weights/weightsRandomTiles.dat")
    else:

        # load image
        model.load_state_dict(torch.load("/gpfs3/well/lindgren/users/swf744/git/pytorch-unet/weights/weightsRandomTiles.dat"))
        model.eval()
        results = predict(model,imageDir,device)

if __name__ == "__main__":
    main()

