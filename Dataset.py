import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import cv2
import os
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
from augment import augmenter

# training is done with a montage
class GetDataTilesArray(Dataset):
    def __init__(self, whichData, preName, pathDir="", transform=None):
        # define the size of the tiles to be working on
        shape = 1024
        assert whichData in ['train', 'validation']            
        files = os.listdir(pathDir)        
        mask_list = []        
        image_list = []        
        self.whichData = whichData
        self.preName = preName
        files.sort()
        self.meanIm=[]
        self.stdIm=[]
        self.epochs=0
        
        for file in files:
            if "_mask.npy" in file:
                continue
            print("file being read is:")
            print(file)
            im = np.load(pathDir + file)
            self.meanIm.append(np.mean(im))
            self.stdIm.append(np.std(im))
            newFile = file.replace(".npy","_mask.npy")
            mask = np.load(pathDir + newFile)
            image_list.append(im)
            mask_list.append(mask)
        
        self.totalMean = sum(self.meanIm) / len(self.meanIm)
        self.totalStd = sum(self.stdIm) / len(self.stdIm)

        f=open("weights/norm"+preName+".norm","w")
        print(self.totalMean,file=f)
        print(self.totalStd,file=f)
        f.close()

        mask_array = np.asarray(mask_list)
        image_array = np.asarray(image_list)
        self.input_images = image_list
        self.target_masks = mask_list
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        i,x,y,first = idx
        shape = 1024
        image = self.input_images[i][x:(x+shape),y:(y+shape)] 
        mask = self.target_masks[i][x:(x+shape),y:(y+shape)] 

        normalize = lambda x: (x - self.totalMean) / (self.totalStd + 1e-10)
        image = normalize(image)
        # augment image 50 % of the time - either rotates or flips 
        image,mask = augmenter(image,mask)

        assert np.shape(image) == (1024,1024)
        assert np.shape(mask) == (1024,1024)

        if first == 1:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_mask.png", mask)
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_mask.png", mask)
            self.epochs += 1

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]

# prediction is done on files in a folder
class GetDataSeqTilesFolder(Dataset):
    def __init__(self, whichData, preName, pathDir="", transform=None):

        # define the size of the tiles to be working on
        shape = 1024
        # so far can only predict with images in a folder
        assert whichData in ['predict']            
        files = os.listdir(pathDir)        
        mask_list = []        
        image_list = []        
        big_mask_list = []        
        big_image_list = []        
        files.sort()
        self.whichData = whichData
        self.counter=0

        f=open("weights/norm"+preName+".norm","r")
        self.totalMean = float(f.readline())
        self.totalStd = float(f.readline())
        f.close()

        for file in files:
            if "_mask.png" in file:
                continue
            print("file being read:")
            print(file)
            im = cv2.imread(pathDir + file, cv2.IMREAD_GRAYSCALE)
            im2 = np.reshape(im,(shape,shape,1))
            normalize = lambda x: (x - self.totalMean) / (self.totalStd + 1e-10)
            mask = np.zeros((shape,shape))                      
            mask_list.append(mask)
            image_list.append(normalize(im2))
            
        self.input_images = image_list
        self.target_masks = mask_list
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
            
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]



