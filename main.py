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

import loss
import Sampler
import Dataset
import prediction
import training
import DataLoader
import model


from model import UNet, UNetnoDial
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
    prs.add_argument('--pathDir', help='path of directory for data', type=str)
    prs.add_argument('--imageDir', help='if directory with images', type=int)
    prs.add_argument('--epochs', help='number of epochs', type=int)
    prs.add_argument('--gamma', help='number of epochs', type=float, default=0)
    prs.add_argument('--weights', help='path to weights', type=str)
    prs.add_argument('--augment', help='whether to augment training', type=int, default=0)
    prs.add_argument('--optimiser', help='which optimiser to use', type=int, default=0)
    prs.add_argument('--dilate', help='to use UNet with dilations or not', type=int, default=1)


    args = vars(prs.parse_args())
    assert args['mode'] in ['train', 'predict']
    assert args['optimiser'] in [0,1]
    assert args['augment'] in [0,1]
    assert args['dilate'] in [0,1]
    assert (args['optimiser'] in [1] and args['gamma']>0) or (args['optimiser'] in [0] and args['gamma']==0)

    assert args['gpu'] in ['0','1','2','3']
    gpu=args['gpu']

    trainOrPredict=args['mode']
    seed=args['seed']
    np.random.seed(seed)
    noTiles=args['tiles']

    pathDir=args['pathDir']
    imageDir=args['imageDir']
    assert imageDir == 0 or imageDir==1
    imageDir=bool(imageDir)

    noEpochs=args['epochs']

    gamma=args['gamma']
    ifAugment=args['augment']
    whichOptim=args['optimiser']


    print("IT IS ASSUMED THAT THIS SCRIPT IS EXECUTED FROM THE DIRECTORY OF THE FILE")

    assert os.path.isfile("main.py") and os.path.isfile("DataLoader.py") and os.path.isfile("loss.py") 

    assert trainOrPredict in ['train', 'predict']

    device = select_gpu(gpu)
    print(device)
        
    num_class = 1

    model = None
    if args['dilate'] == 1:
        model = UNet(n_class=num_class).to(device)
    else:
        model = UNetnoDial(n_class=num_class).to(device)

    #model.double()
    print(model)
    
    print("N parameters:")
    print(count_parameters(model))

    date = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    
    randOrSeq = ""
    if imageDir:
        randOrSeq = "Seq"
    else:
        randOrSeq = "Random"

    preName = randOrSeq+"Tiles_epochs"+str(noEpochs)+"time"+date+"gamma"+str(gamma)+"seed"+str(seed)+"aug"+str(ifAugment)+"optim"+str(whichOptim)

    if(trainOrPredict == "train"):
        training_data = get_dataloader(pathDir,imageDir,preName,ifAugment,noTiles)

        if whichOptim==0:

            optimizer_ft = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
            # same scheduler as craig is using 
            lr_scheduler1 = lr_scheduler.CyclicLR(optimizer_ft, base_lr=0.00001, max_lr=0.0005, mode='triangular')

        else:
            # original scheduler, gives better performances apparently!?            
            optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
            lr_scheduler1 = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=float(gamma))


        os.mkdir('crops'+preName+'/') 

        f=open("log"+preName+".log","w")
        model = train_model(model, training_data, device, optimizer_ft, lr_scheduler1, f, preName, num_epochs=noEpochs)
        if os.path.isdir('weights/'): 
            torch.save(model.state_dict(),"weights/weights"+preName+".dat")
        else:
            os.mkdir('weights/') 
            torch.save(model.state_dict(),"weights/weights"+preName+".dat")
        f.close()
    else:
        predictWeights= sys.argv[8] 
        preName = predictWeights.replace("weights/weights","")
        preName = preName.replace(".dat","")
        # load image
        model.load_state_dict(torch.load(predictWeights))
        model.eval()
        results = predict(model,pathDir,imageDir,device,preName)

if __name__ == "__main__":
    main()

