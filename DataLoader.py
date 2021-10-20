import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from torch.utils.data.sampler import Sampler

from Dataset import GetData
from Sampler import RandomSampler

def get_dataloader():

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    # read in shuffled data
    train_set = GetData("train", transform=trans)
    val_set = GetData("validation", transform=trans)
    
    image_datasets = {
        'train': train_set, 'val': val_set
    }

    sample_size_train = 200
    sample_size_val = 20
    batch_size = 2
    
    samplie_train = RandomSampler(train_set, sample_size_train, 1024, False, 0)
    samplie_val = RandomSampler(val_set, sample_size_val, 1024, True, 0)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
        'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
    }
    
    return dataloaders

