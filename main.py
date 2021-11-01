#from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
import os
from torch.utils.data.sampler import Sampler
import numpy as np
from datetime import datetime

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
    
    print("len is:")
    print(len(sys.argv))

    ## index, 0 will give you the python filename being executed. Any index after that are the arguments passed.
    if(len(sys.argv)==1):
        print("1. which GPU (int), 2. 'train'/'predict', 3. seed (int), 4. path of directory for data (string), 5. if directory with images (boolean), 6 epochs (int - only for training)")
        print("7. gamma (for decaying learning rate - only for training) , 8. path of weights (only for predicting)")
        sys.exit("")

    gpu=sys.argv[1]
    trainOrPredict=sys.argv[2]
    seed=int(sys.argv[3])
    np.random.seed(seed)

    pathDir=sys.argv[4]
    imageDir=int(sys.argv[5])
    assert imageDir == 0 or imageDir==1
    imageDir=bool(imageDir)

    noEpochs=int(sys.argv[6])

    gamma=sys.argv[7]


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

    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    if(trainOrPredict == "train"):
        training_data = get_dataloader(pathDir,imageDir)
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=float(gamma))

        f=open("log_epochs"+str(noEpochs)+"time"+date+"gamma"+gamma+".log","w")
        model = train_model(model, training_data, device, optimizer_ft, exp_lr_scheduler, f, num_epochs=noEpochs)
        if os.path.isdir('weights/'): 
            torch.save(model.state_dict(),"weights/weightsRandomTiles.dat")
            torch.save(model.state_dict(),"weights/weightsRandomTiles_epochs"+str(noEpochs)+"time"+date+"gamma"+gamma+".dat")
        else:
            os.mkdir('weights/') 
            torch.save(model.state_dict(),"weights/weightsRandomTiles.dat")
            torch.save(model.state_dict(),"weights/weightsRandomTiles_epochs"+str(noEpochs)+"time"+date+"gamma"+gamma+".dat")
        f.close()
    else:
        predictWeights= sys.argv[8] 
        # load image
        model.load_state_dict(torch.load(predictWeights))
        model.eval()
        results = predict(model,pathDir,imageDir,device)

if __name__ == "__main__":
    main()

