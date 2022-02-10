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
import json
from augment import augmenter, albumentationAugmenter

# training is done with a montage
class GetDataTilesArray(Dataset):
    def __init__(self, whichData, preName, augSeed, frank, pathDir="", transform=None, ifAugment=0):
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
        self.ifAugment=ifAugment
        self.frank=frank
        self.augSeed=augSeed
                
        for file in files:
            if "_mask.npy" in file:
                continue
            print("file being read is:")
            print(file)
            im = np.load(pathDir + file)
            self.meanIm.append(np.mean(im))
            self.stdIm.append(np.std(im))
            newFile = file.replace(".npy","_mask.npy")
            #because empty spaces are 1 and adipocytes 0 originally
            mask = 1-np.load(pathDir + newFile)
            image_list.append(im)
            mask_list.append(mask)

        # should normalise with training data
        if self.whichData == "train":
            self.totalMean = sum(self.meanIm) / len(self.meanIm)
            self.totalStd = sum(self.stdIm) / len(self.stdIm)
            f=open("weights/norm"+preName+".norm","w")
            print(self.totalMean,file=f)
            print(self.totalStd,file=f)
            f.close()
        # with val data it should read data from training data
        else:
            f=open("weights/norm"+preName+".norm","r")
            self.totalMean = float(f.readline())
            self.totalStd = float(f.readline())
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

        if self.whichData=="train" and self.frank == 1:
            
            # unpacking coordinate lists (xlist and ylist) and where to cut x and y (cutx and cuty)
            xlist=x[0]
            ylist=x[1]
            cutx=y[0]
            cuty=y[1]
            
            # upper left corner, upper right corner, lower left corner, lower right corner
            image1 = self.input_images[i[0]][xlist[0]:(xlist[0]+cutx),ylist[0]:(ylist[0]+cuty)] 
            image2 = self.input_images[i[1]][xlist[1]:(xlist[1]+cutx),(ylist[1]+cuty):(ylist[1]+shape)] 
            image3 = self.input_images[i[2]][(xlist[2]+cutx):(xlist[2]+shape),ylist[2]:(ylist[2]+cuty)] 
            image4 = self.input_images[i[3]][(xlist[3]+cutx):(xlist[3]+shape),(ylist[3]+cuty):(ylist[3]+shape)] 

            # concat upper and lower parts (add columns together so has more columns now)
            imageCat = np.concatenate((image1,image2),axis=1)
            imageCat2 = np.concatenate((image3,image4),axis=1)
            # concat upper and lower part (add rows together so has more rows now)
            image = np.concatenate((imageCat,imageCat2),axis=0)
            
            # upper left corner, upper right corner, lower left corner, lower right corner
            mask1 = self.target_masks[i[0]][xlist[0]:(xlist[0]+cutx),ylist[0]:(ylist[0]+cuty)] 
            mask2 = self.target_masks[i[1]][xlist[1]:(xlist[1]+cutx),(ylist[1]+cuty):(ylist[1]+shape)] 
            mask3 = self.target_masks[i[2]][(xlist[2]+cutx):(xlist[2]+shape),ylist[2]:(ylist[2]+cuty)] 
            mask4 = self.target_masks[i[3]][(xlist[3]+cutx):(xlist[3]+shape),(ylist[3]+cuty):(ylist[3]+shape)] 
            maskCat = np.concatenate((mask1,mask2),axis=1)
            maskCat2 = np.concatenate((mask3,mask4),axis=1)
            mask = np.concatenate((maskCat,maskCat2),axis=0)

        else:
            image = self.input_images[i][x:(x+shape),y:(y+shape)] 
            mask = self.target_masks[i][x:(x+shape),y:(y+shape)] 
        choice=0

        if first == 1:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_mask.png", mask)
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_mask.png", mask)

        # only augments training images - does 50 % of the time - rotates, flips, blur or noise
        if self.whichData=="train" and self.ifAugment:
            #emil convert to uint8 instead of float32 - might cause issues
            #image = image.astype(np.uint8, copy=False)
            #mask = mask.astype(np.uint8, copy=False)
            #because gaussNoise and RandomBrightness only made for floats between 0 and 1
            image = image/255.0
            image,mask,replay,choice,crop = albumentationAugmenter(image,mask,self.epochs)
            #and convert it back to float32 - because rest of pipeline built for that
            #image = image.astype(np.float32, copy=False)
            #mask = mask.astype(np.float32, copy=False)
            image = image*255.0

        # to pad with zeros
        if(np.shape(image)<(1024,1024)):
            image=np.pad(image, [(1024-crop)//2, (1024-crop)//2], mode='constant')
            mask=np.pad(mask, [(1024-crop)//2, (1024-crop)//2], mode='constant')

        assert np.shape(image) == (1024,1024)
        assert np.shape(mask) == (1024,1024)

        if choice > 0 and first == 1:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+"_mask.png", mask)
                with open("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+"_whichAlbu.txt", 'w') as f:
                    print(replay, file=f)
                f.close()                
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+str(i)+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+"_mask.png", mask)
                
        if first == 1:
            self.epochs += 1

        normalize = lambda x: (x - self.totalMean) / (self.totalStd + 1e-10)
        image = normalize(image)
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
