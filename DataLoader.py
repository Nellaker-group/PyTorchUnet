import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from torch.utils.data.sampler import Sampler
import os

from Dataset import GetDataTilesArray, GetDataSeqTilesFolder
from Sampler import RandomSampler, SeqSamplerDatasetSize, SeqSamplerUniform, ValSampler


def get_dataloader(pathDir,imageDir,preName,ifAugment,noTiles):

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainPathDir=pathDir + "train/"
    valPathDir=pathDir + "val/"

    print("imageDir:")
    print(imageDir)

    # read in data
    train_set = GetDataTilesArray("train", preName, pathDir=trainPathDir, transform=trans, ifAugment=ifAugment)
    val_set = GetDataTilesArray("validation", preName, pathDir=valPathDir, transform=trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }
    
    sample_size_train = noTiles
    # it uses all val tiles
    batch_size = 2

    dataloaders = {}

    if imageDir:
        # read in data
        samplie_train = SeqSamplerDatasetSize(train_set, sample_size_train, 1024, 0)
        samplie_val = ValSampler(val_set,  1024, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }
    else:
        # read in data
        samplie_train = RandomSampler(train_set, sample_size_train, 1024, 0)
        samplie_val = ValSampler(val_set, 1024, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }


    return dataloaders

