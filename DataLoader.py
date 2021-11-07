import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from torch.utils.data.sampler import Sampler
import os

from Dataset import GetDataSeqTilesArray, GetDataTilesArray, GetDataSeqTilesFolder
from Sampler import RandomSampler, SeqSampler, SeqSamplerV2

def get_dataloader(pathDir,imageDir,preName):

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    trainPathDir=pathDir + "train/"
    valPathDir=pathDir + "val/"

    print("imageDir:")
    print(imageDir)

    files = os.listdir(trainPathDir)
    npy = False
    for file in files:
        npy = file.endswith("npy")
    
    if imageDir and npy:
        # read in data
        # Emil get Random tiles is used here
        train_set = GetDataTilesArray("train", preName, pathDir=trainPathDir, transform=trans)
        val_set = GetDataTilesArray("validation", preName, pathDir=valPathDir, transform=trans)
    elif imageDir:
        # read in data
        train_set = GetDataSeqTilesFolder("train", preName, pathDir=trainPathDir, transform=trans)
        val_set = GetDataSeqTilesFolder("validation", preName, pathDir=valPathDir, transform=trans)
    else:
        # read in data
        train_set = GetDataTilesArray("train", preName, pathDir=trainPathDir, transform=trans)
        val_set = GetDataTilesArray("validation", preName, pathDir=valPathDir, transform=trans)
    
    image_datasets = {
        'train': train_set, 'val': val_set
    }

    sample_size_train = 1559
    sample_size_val = 73
    batch_size = 2

    dataloaders = {}

    if imageDir and npy:
        # read in data
        samplie_train = SeqSamplerV2(train_set, sample_size_train, 1024, 0)
        samplie_val = SeqSamplerV2(val_set, sample_size_val, 1024, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }

    elif imageDir:
        # read in data
        samplie_train = SeqSampler(train_set, sample_size_train, 1024, 0)
        samplie_val = SeqSampler(val_set, sample_size_val, 1024, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }

    else:
        # read in data
        samplie_train = RandomSampler(train_set, sample_size_train, 1024, 0)
        samplie_val = RandomSampler(val_set, sample_size_val, 1024, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }


    

    
    return dataloaders

