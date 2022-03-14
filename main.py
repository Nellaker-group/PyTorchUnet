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
import argparse
import random

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

    prs = argparse.ArgumentParser()
    prs.add_argument('--gpu', help='which GPU to run on', type=str)
    prs.add_argument('--mode', help='train or predict', type=str)
    prs.add_argument('--tiles', help='how many tiles to use for training', type=int, default=200)
    prs.add_argument('--seed', help='seed to use', type=int)
    prs.add_argument('--trainDir', help='path of directory for training data', type=str)
    prs.add_argument('--valDir', help='path of directory for validation data', type=str)
    prs.add_argument('--imageDir', help='if training data is directory with images', type=int)
    prs.add_argument('--epochs', help='number of epochs', type=int)
    prs.add_argument('--gamma', help='number of epochs', type=float, default=0)
    prs.add_argument('--weights', help='path to weights', type=str)
    prs.add_argument('--augment', help='whether to augment training', type=int, default=0)
    prs.add_argument('--optimiser', help='which optimiser to use, (cyclicLR=0, stepLR=1)', type=int, default=0)
    prs.add_argument('--stepSize', help='which step size to use for stepLR optimiser (--optimiser 1)', type=int, default=0)
    prs.add_argument('--torchSeed', help='seed for PyTorch so can control initialization  of weights', type=int, default=0)
    prs.add_argument('--frankenstein', help='assembles tiles from 4 different parts from different tiles (works for montages and uniform sampling across datasets)\n 1=Cuts 4 random parts from tiles and merges them together\n 2=Cuts 4 corners from tiles from the same dataset and merges them together', type=int, default=0)
    prs.add_argument('--sizeBasedSamp', help='if sampling from datasets should depend on the size of the datasets (yes=1, no=0)', type=int, default=0)
    prs.add_argument('--LR', help='start learning rate', type=float)
    prs.add_argument('--inputChannels', help='number of input channels - only works for values != 1 with --imageDir 1', type=int, default=1)
    prs.add_argument('--outputChannels', help='number of output channels or classes to predict', type=int, default=2)
    prs.add_argument('--normFile', help='file with mean and SD for normalisation (1st line mean, 2nd line SD)', type=str)
    prs.add_argument('--512', help='image size is 512, cuts existing 1024x1024 tiles into 4 bits', type=int, default=0)

    args = vars(prs.parse_args())
    assert args['mode'] in ['train', 'predict']
    assert args['optimiser'] in [0,1]
    assert args['augment'] in [0,1]
    assert args['frankenstein'] in [0,1,2]
    assert (args['optimiser'] in [1] and args['gamma']>0) or (args['optimiser'] in [0] and args['gamma']==0)
    assert (args['optimiser'] in [1] and args['stepSize']>0) or (args['optimiser'] in [0] and args['stepSize']==0)
    assert args['sizeBasedSamp'] in [0,1]
    assert args['inputChannels'] in [1] or (args['inputChannels'] in [3] and args['imageDir'] in [1])
    assert args['512'] in [0,1]

    assert args['gpu'] in ['0','1','2','3']
    gpu=args['gpu']

    trainOrPredict=args['mode']
    seed=args['seed']
    np.random.seed(seed)
    noTiles=args['tiles']

    trainDir=args['trainDir']
    valDir=args['valDir']
    imageDir=args['imageDir']
    normFile=args['normFile']
    assert imageDir == 0 or imageDir==1
    imageDir=bool(imageDir)

    noEpochs=args['epochs']
    stepSize=args['stepSize']
    gamma=args['gamma']
    ifAugment=args['augment']
    whichOptim=args['optimiser']
    ifSizeBased=args['sizeBasedSamp']
    learningRate=args['LR']
    frank=args['frankenstein']
    input512=args['512']
    
    inputSize = 1024
    if input512 == 1:
        inputSize = 512

    if args['torchSeed']>0:
        torch.manual_seed(args['torchSeed'])
        print("TORCH SEED IS: "+str(args['torchSeed']))

    print("IT IS ASSUMED THAT THIS SCRIPT IS EXECUTED FROM THE DIRECTORY OF THE FILE")

    assert os.path.isfile("main.py") and os.path.isfile("DataLoader.py") and os.path.isfile("loss.py") 

    assert trainOrPredict in ['train', 'predict']

    #device = torch.device("cpu")
    device = select_gpu(gpu)
    print(device)
        
    num_class = args['outputChannels'] - 1
    inputChannels = args['inputChannels']

    model = None
    if inputChannels == 1:
        model = UNet(n_class=num_class,n_input=inputChannels).to(device)
    else:
        model = UNet(n_class=num_class, n_input=inputChannels).to(device)

    print("weights are:")
    for param in model.parameters():
        print(param[0:10])
        break
    
    print(model)
    
    print("N parameters:")
    print(count_parameters(model))

    date = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    
    randOrSeq = ""
    if imageDir:
        randOrSeq = "Seq"
    else:
        randOrSeq = "Random"

    preName = randOrSeq+"Tiles_ep"+str(noEpochs)+"t"+date+"g"+str(gamma)+"s"+str(seed)+"au"+str(ifAugment)+"op"+str(whichOptim)+"st"+str(stepSize)+"sB"+str(ifSizeBased)+"LR"+str(learningRate)+"fr"+str(frank)+"ch"+str(inputChannels)+"si"+str(inputSize)
    # for the sampling of the augmentation
    augSeed = np.random.randint(0,100000)
    random.seed(augSeed)

    if(trainOrPredict == "train"):
        training_data = get_dataloader(trainDir, valDir,imageDir,preName,ifAugment,noTiles,augSeed,ifSizeBased,frank,inputChannels,normFile,input512)

        if whichOptim==0:

            optimizer_ft = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learningRate)
            # same scheduler as craig is using 
            lr_scheduler1 = lr_scheduler.CyclicLR(optimizer_ft, base_lr=(learningRate/10), max_lr=(learningRate*5), mode='triangular')

        else:
            # original scheduler, gives better performances apparently!?            
            optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningRate)
            lr_scheduler1 = lr_scheduler.StepLR(optimizer_ft, step_size=stepSize, gamma=float(gamma))
            
        os.mkdir('crops'+preName+'/') 

        f=open("log"+preName+".log","w")
        model = train_model(model, training_data, device, optimizer_ft, lr_scheduler1, f, preName, whichOptim, num_epochs=noEpochs)
        if os.path.isdir('weights/'): 
            torch.save(model.state_dict(),"weights/weights"+preName+".dat")
        else:
            os.mkdir('weights/') 
            torch.save(model.state_dict(),"weights/weights"+preName+".dat")
        f.close()
    else:
        predictWeights=args['weights']
        preName = predictWeights.replace("weights/weights","")
        preName = preName.replace(".dat","")
        # load image
        model.load_state_dict(torch.load(predictWeights))
        model.eval()
        results = predict(model,pathDir,imageDir,device,preName)

if __name__ == "__main__":
    main()

