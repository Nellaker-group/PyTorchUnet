import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import cv2
import os
import sys
import math

# Emil
import matplotlib
import matplotlib.pyplot as plt


class GetDataTilesArray(Dataset):

    def __init__(self, whichData, preName, pathDir="", transform=None):

        # define the size of the tiles to be working on
        shape = 1024
        assert whichData in ['train', 'validation', 'predict']            
        files = os.listdir(pathDir)        
        mask_list = []        
        image_list = []        
        self.whichData = whichData
        self.preName = preName
        files.sort()
        self.counter=0
    
        for file in files:
            if "_mask.npy" in file:
                continue
            print("file being read is:")
            print(file)
            im = np.load(pathDir + file)
            if(self.whichData=="predict"):
                meanIm, stdIm = np.mean(im), np.std(im)
                normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)
                image_list.append(normalize(im))
                mask = np.zeros((shape,shape))          
                mask_list.append(mask)
            else:
                newFile = file.replace(".npy","_mask.npy")
                mask = np.load(pathDir + newFile)
                image_list.append(im)
                mask_list.append(mask)

        mask_array = np.asarray(mask_list)
        image_array = np.asarray(image_list)
        self.input_images = image_list
        self.target_masks = mask_list
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        i,x,y = idx
        shape = 1024
        image = self.input_images[i][x:(x+shape),y:(y+shape)] 
        mask = self.target_masks[i][x:(x+shape),y:(y+shape)] 

        assert np.shape(image) == (1024,1024)
        assert np.shape(mask) == (1024,1024)

        if self.counter < 20:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_"+str(i)+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/train_"+str(i)+"_"+str(x)+"_"+str(y)+"_mask.png", mask)
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_"+str(i)+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/val_"+str(i)+"_"+str(x)+"_"+str(y)+"_mask.png", mask)

        self.counter += 1

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]


class GetDataSeqTilesArray(Dataset):

    def __init__(self, whichData, pathDir="", transform=None):
        # define the size of the tiles to be working on
        shape = 1024
        assert whichData in ['train', 'validation', 'predict']            
        files = os.listdir(pathDir)        
        mask_list = []        
        image_list = []        
        big_mask_list = []
        big_image_list = []
        files.sort()

        count = 0
        # check that there is an even number of files in the folder
        assert len(files) % 2 == 0

        # I assume half of the files are masks and the other half images
        image_array = np.zeros((len(files)//2))
        mask_array = np.zeros((len(files)//2))
        self.whichData = whichData
        self.counter=0
    
        for file in files:
            if "_mask.npy" in file:
                continue
            print("file being read is:")
            print(file)
            im = np.load(pathDir + file)
            if(whichData!="predict"):
                newFile = file.replace(".npy","_mask.npy")
                mask = np.load(pathDir + newFile)

            rows,cols = np.shape(im)
            meanIm, stdIm = np.mean(im), np.std(im)
            normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)
            # getting the number of tiles the image contains
            tiles = math.floor(rows // shape) * math.floor(cols // shape)

            # loop over the image and put the image into a numpy ndarray
            # each image is a (1024x1024x3) numpy array, where 3 is the number of colour channels
            for x in range(0, rows, shape):
                for y in range(0, cols, shape):
                    if((x+shape)>rows or (y+shape)>cols):
                        continue
                    # reading it in row by row, so start top left then move right, and then down and all over
                    if(whichData=="predict"):
                        image_list.append(normalize(im[x:(x+shape),y:(y+shape),]))
                        mask = np.zeros((shape,shape))          
                        mask_list.append(mask)
                    else:
                        image_list.append(im[x:(x+shape),y:(y+shape),])
                        mask_list.append(mask[x:(x+shape),y:(y+shape),])
            big_mask_list.append(np.asarray(mask_list))
            big_image_list.append(np.asarray(image_list))

        self.input_images = big_image_list
        self.target_masks = big_mask_list
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        i,j = idx
        image = self.input_images[i][j]
        mask = self.target_masks[i][j]

        assert np.shape(image) == (1024,1024)
        assert np.shape(mask) == (1024,1024)

        if self.counter < 20:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_"+str(i)+"_"+str(j)+".png", image)
                plt.imsave("crops"+self.preName+"/train_"+str(i)+"_"+str(j)+"_mask.png", mask)
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_"+str(i)+"_"+str(j)+".png", image)
                plt.imsave("crops"+self.preName+"/val_"+str(i)+"_"+str(j)+"_mask.png", mask)

        self.counter += 1

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]


class GetDataSeqTilesFolder(Dataset):

    def __init__(self, whichData, pathDir="", transform=None):

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

        for file in files:
            if "_mask.png" in file:
                continue
            im = cv2.imread(pathDir + file, cv2.IMREAD_GRAYSCALE)
            im2 = np.reshape(im,(shape,shape,1))
            if(whichData=="predict"):
                meanIm, stdIm = np.mean(im2), np.std(im2)
                normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)
                mask = np.zeros((shape,shape))                      
                mask_list.append(mask)
                image_list.append(normalize(im2))
            else:
                newFile = file.replace(".png","_mask.png")
                newFile = newFile.replace(".jpg","_mask.png")
                # assumed mask file have same name as image files, but just end with *_mask.png or *_mask.jpg  
                mask = cv2.imread(pathDir + newFile, cv2.IMREAD_GRAYSCALE)
                mask2 = np.reshape(mask,(shape,shape,1))
                mask_list.append(mask2)
                # loop over the image and put the image into a numpy ndarray                                                                
                # each image is a (1024x1024x3) numpy array, where 3 is the number of colour channels    
                image_list.append(im2)

        self.input_images = image_list
        self.target_masks = mask_list
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]

        if self.counter < 10 and (self.whichData=="train" or self.whichData=="validation"):
            print("HERE:")
            print("cropsFolder/"+str(idx)+".png")

            plt.imsave("cropsFolder/"+str(idx)+".png", image)
            plt.imsave("cropsFolder/"+str(idx)+"_mask.png", mask)
            self.counter+=1
            
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]



