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
class GetDataMontage(Dataset):
    def __init__(self, whichData, preName, augSeed, frank, normFile, pathDir="", transform=None, ifAugment=0):
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

        # should normalise with data from normFile
        f=open(normFile,"r")
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
        data,x,y,first = idx
        shape = 1024

        dataName = str(data)

        if self.whichData=="train" and self.frank == 1:
            
            # unpacking coordinate lists (xlist and ylist) and where to cut x and y (cutx and cuty)
            xlist=x[0]
            ylist=x[1]
            cutx=y[0]
            cuty=y[1]
            
            # upper left corner, upper right corner, lower left corner, lower right corner
            image1 = self.input_images[data[0]][xlist[0]:(xlist[0]+cutx),ylist[0]:(ylist[0]+cuty)] 
            image2 = self.input_images[data[1]][xlist[1]:(xlist[1]+cutx),(ylist[1]+cuty):(ylist[1]+shape)] 
            image3 = self.input_images[data[2]][(xlist[2]+cutx):(xlist[2]+shape),ylist[2]:(ylist[2]+cuty)] 
            image4 = self.input_images[data[3]][(xlist[3]+cutx):(xlist[3]+shape),(ylist[3]+cuty):(ylist[3]+shape)] 

            # concat upper and lower parts (add columns together so has more columns now)
            imageCat = np.concatenate((image1,image2),axis=1)
            imageCat2 = np.concatenate((image3,image4),axis=1)
            # concat upper and lower part (add rows together so has more rows now)
            image = np.concatenate((imageCat,imageCat2),axis=0)
            
            # upper left corner, upper right corner, lower left corner, lower right corner
            mask1 = self.target_masks[data[0]][xlist[0]:(xlist[0]+cutx),ylist[0]:(ylist[0]+cuty)] 
            mask2 = self.target_masks[data[1]][xlist[1]:(xlist[1]+cutx),(ylist[1]+cuty):(ylist[1]+shape)] 
            mask3 = self.target_masks[data[2]][(xlist[2]+cutx):(xlist[2]+shape),ylist[2]:(ylist[2]+cuty)] 
            mask4 = self.target_masks[data[3]][(xlist[3]+cutx):(xlist[3]+shape),(ylist[3]+cuty):(ylist[3]+shape)] 
            maskCat = np.concatenate((mask1,mask2),axis=1)
            maskCat2 = np.concatenate((mask3,mask4),axis=1)
            mask = np.concatenate((maskCat,maskCat2),axis=0)

            dataName = dataName.replace("[","").replace("]","").replace("(","").replace(")","").replace(", ","-")

        else:
            image = self.input_images[data][x:(x+shape),y:(y+shape)] 
            mask = self.target_masks[data][x:(x+shape),y:(y+shape)] 
        choice=0

        if first == 1:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_mask.png", mask)
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_mask.png", mask)

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
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+"_mask.png", mask)
                with open("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+"_whichAlbu.txt", 'w') as f:
                    print(replay, file=f)
                f.close()                
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_"+str(x)+"_"+str(y)+"_albuChoice"+str(choice)+"_mask.png", mask)
                
        if first == 1:
            self.epochs += 1

        normalize = lambda x: (x - self.totalMean) / (self.totalStd + 1e-10)
        image = normalize(image)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]

# training is done here
class GetDataFolder(Dataset):
    def __init__(self, whichData, preName, augSeed, frank, inputChannels, normFile, pathDir="", transform=None, ifAugment=0):
        # define the size of the tiles to be working on
        shape = 1024
        # so far can only predict with images in a folder
        assert whichData in ['train','validation']            
        directories = os.listdir(pathDir)        
        mask_list = []        
        image_list = []        
        directories.sort()
        self.whichData = whichData
        self.preName = preName
        self.meanIm=[]
        self.stdIm=[]
        self.epochs=0
        self.ifAugment=ifAugment
        self.frank=frank
        self.augSeed=augSeed
        self.whichData = whichData
        self.datasetMean=[]
        self.datasetStd=[]
        self.inputChannels=inputChannels

        whichFolder = 0

        for directory in directories:
            files = os.listdir(pathDir + "/" + directory)        
            image_list.append([])
            mask_list.append([])
            for file in files:
                if "mask_" in file and (".png" in file or ".jpg" in file):
                    continue
                if not (".png" in file or ".jpg" in file):
                    continue
                if inputChannels == 1:
                    im = cv2.imread(pathDir + "/" + directory + "/" + file, cv2.IMREAD_GRAYSCALE)
                    assert np.shape(im) != ()
                    im = im.astype(np.float32)
                    if(np.shape(im)>(1024,1024)):
                        im = im[0:1024,0:1024]
                else:
                    im = cv2.imread(pathDir + "/" + directory + "/" + file)
                    assert np.shape(im) != ()
                    im = im.astype(np.float32)
                    if(np.shape(im)>(1024,1024,3)):
                        im = im[0:1024,0:1024,0:3]
                mask = cv2.imread(pathDir + "/" + directory + "/mask_" + file, cv2.IMREAD_GRAYSCALE)
                assert np.shape(mask) != ()
                assert np.max(mask) == 255
                assert np.min(mask) == 0
                # because this is not a binary mask for some reason - converting it into a binary mask
                middlePoint = (np.max(mask)+np.min(mask))/2
                maskCopy = np.copy(mask)
                mask[ maskCopy < middlePoint ]=1
                mask[ maskCopy > middlePoint ]=0
                mask = mask.astype(np.float32)
                if(np.shape(mask)>(1024,1024)):
                    mask = mask[0:1024,0:1024]
                image_list[whichFolder].append(im)
                mask_list[whichFolder].append(mask)
                self.meanIm.append(np.mean(im))
                self.stdIm.append(np.std(im))
            whichFolder += 1
            self.datasetMean.append(np.mean(self.meanIm))
            self.datasetStd.append(np.mean(self.stdIm))
            self.meanIm=[]
            self.stdIm=[]

        # should normalise with data from normFile
        f=open(normFile,"r")
        self.totalMean = float(f.readline())
        self.totalStd = float(f.readline())
        f.close()

        assert len(image_list) == len(mask_list)

        for item in range(len(image_list)):
            mask_list[item] = np.asarray(mask_list[item])
            image_list[item] = np.asarray(image_list[item])

        mask_array = np.asarray(mask_list)
        image_array = np.asarray(image_list)

        self.input_images = image_list
        self.target_masks = mask_list
        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        data,i,first = idx
        shape = 1024

        dataName = str(data)+"_"+str(i)

        if self.whichData=="train" and self.frank == 1:

            dataName = str(data)+"_"+str(i[0])
            
            # unpacking coordinate lists (xlist and ylist) and where to cut x and y (cutx and cuty)
            indexes = i[0]
            cuts = i[1]
            x0=cuts[0]
            y0=cuts[1]
            x1=cuts[2]
            y1=cuts[3]
            x2=cuts[4]
            y2=cuts[5]
            x3=cuts[6]
            y3=cuts[7]
            h=cuts[8]
            w=cuts[9]
            w1=cuts[10]            

            # upper left corner, upper right corner, lower left corner, lower right corner
            image1 = self.input_images[data[0]][indexes[0]][x0:(x0+h),y0:(y0+w)]
            image2 = self.input_images[data[1]][indexes[1]][x1:(x1+h),y1:(y1+(1024-w))]
            image3 = self.input_images[data[2]][indexes[2]][x2:(x2+(1024-h)),y2:(y2+w1)]
            image4 = self.input_images[data[3]][indexes[3]][x3:(x3+(1024-h)),y3:(y3+(1024-w1))]

            # concat upper and lower parts (add columns together so has more columns now)
            imageCat = np.concatenate((image1,image2),axis=1)
            imageCat2 = np.concatenate((image3,image4),axis=1)
            # concat upper and lower part (add rows together so has more rows now)
            image = np.concatenate((imageCat,imageCat2),axis=0)
            
            # upper left corner, upper right corner, lower left corner, lower right corner
            mask1 = self.target_masks[data[0]][indexes[0]][x0:(x0+h),y0:(y0+w)]
            mask2 = self.target_masks[data[1]][indexes[1]][x1:(x1+h),y1:(y1+(1024-w))]
            mask3 = self.target_masks[data[2]][indexes[2]][x2:(x2+(1024-h)),y2:(y2+w1)]
            mask4 = self.target_masks[data[3]][indexes[3]][x3:(x3+(1024-h)),y3:(y3+(1024-w1))]
            maskCat = np.concatenate((mask1,mask2),axis=1)
            maskCat2 = np.concatenate((mask3,mask4),axis=1)
            mask = np.concatenate((maskCat,maskCat2),axis=0)

            dataName = dataName.replace("[","").replace("]","").replace("(","").replace(")","").replace(", ","-")

        else:
            image = self.input_images[data][i] 
            mask = self.target_masks[data][i] 
        choice=0

        if self.inputChannels == 3:
            #this makes the image being print correctly and is needed for the albumentations augs
            image = cv2.cvtColor(image/255.0, cv2.COLOR_RGB2BGR)

        if first == 1:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_mask.png", mask)
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_mask.png", mask)

        if self.inputChannels == 3:
            #this makes the image being print correctly
            image = image*255

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

        if self.inputChannels == 3:
            #this makes the image being print correctly
            image = image/255.0
            assert np.shape(image) == (1024,1024,3)
            assert np.shape(mask) == (1024,1024)
        else:
            assert np.shape(image) == (1024,1024)
            assert np.shape(mask) == (1024,1024)

        if choice > 0 and first == 1:
            # crops from first run
            if self.whichData=="train":
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_albuChoice"+str(choice)+".png", image)
                plt.imsave("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_albuChoice"+str(choice)+"_mask.png", mask)
                with open("crops"+self.preName+"/train_epochs"+str(self.epochs)+"_"+dataName+"_albuChoice"+str(choice)+"_whichAlbu.txt", 'w') as f:
                    print(replay, file=f)
                f.close()                
            elif self.whichData=="validation":
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_albuChoice"+str(choice)+".png", image)
                plt.imsave("crops"+self.preName+"/val_epochs"+str(self.epochs)+"_"+dataName+"_albuChoice"+str(choice)+"_mask.png", mask)

        if self.inputChannels == 3:
            image = image*255.0
                        
        if first == 1:
            self.epochs += 1

        normalize = lambda x: (x - self.totalMean) / (self.totalStd + 1e-10)
        image = normalize(image)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return [image, mask]
            
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]



# prediction is done on files in a folder
class GetDataSeqTilesFolderPred(Dataset):
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
