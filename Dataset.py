import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import cv2
import os

class GetDataRandomTiles(Dataset):

    def __init__(self, whichData, pathDir="", transform=None):

        # define the size of the tiles to be working on
        shape = 1024
        assert whichData in ['train', 'validation', 'predict']            
        files = os.listdir(pathDir)        
        image_array = np.zeros((len(files),shape,shape,1))
        mask_list = []        
        image_list = []        
        count = 0
    
        for file in files:
            if "_mask.npy" in file:
                continue
            im = np.load(pathDir + file)
            image_list.append(im)
            if(whichData=="predict"):
                meanIm, stdIm = np.mean(im), np.std(im)
                normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)
                mask = np.zeros((shape,shape))          
                mask_list.append(mask)
            else:
                newFile = file.replace(".npy","_mask.npy")
                mask = np.load(pathDir + newFile)
                mask_list.append(mask)
        mask_array = np.asarray(mask_list)
        image_array = np.asarray(image_list)

        self.input_images = image_array
        self.target_masks = mask_array
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        i,x,y = idx
        shape = 1024

        image = self.input_images[i][x:(x+shape),y:(y+shape)] 
        mask = self.target_masks[i][x:(x+shape),y:(y+shape)] 

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]



class GetDataSeqTiles(Dataset):

    def __init__(self, whichData, pathDir="", transform=None):

        # define the size of the tiles to be working on
        shape = 1024
        assert whichData in ['train', 'validation', 'predict']            
        files = os.listdir(pathDir)        
        image_array = np.zeros((len(files),shape,shape,1))
        mask_list = []        
        image_list = []        
        count = 0
    
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
            else:
                newFile = file.replace(".png","_mask.png")
                newFile = newFile.replace(".jpg","_mask.png")
                # assumed mask file have same name as image files, but just end with *_mask.png or *_mask.jpg  
                mask = cv2.imread(pathDir + newFile, cv2.IMREAD_GRAYSCALE)
                mask2 = np.reshape(mask,(shape,shape,1))
                mask_list.append(mask2)
                # loop over the image and put the image into a numpy ndarray                                                                
                # each image is a (1024x1024x3) numpy array, where 3 is the number of colour channels    

            print("im2:")
            print(im2)
            image_array[count,0:shape,0:shape,] = normalize(im2)
            count += 1

        mask_array = np.asarray(mask_list)
    
        self.input_images = image_array
        self.target_masks = mask_array
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



