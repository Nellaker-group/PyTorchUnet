import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import cv2

class GetData(Dataset):
    def __init__(self, whichData, imageDir="", transform=None):

        # define the size of the tiles to be working on
        shape = 1024

        assert whichData in ['train', 'validation', 'predict']

        if(whichData=="validation"):

            # we read the .png image into python as a numpy ndarray using the cv2.imread function
            im = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_img2_val.npy')
            im2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_img2_val.npy')
            im3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_img2_val.npy')
            im4 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_img2_val.npy')

            self.input_images = np.array([im, im2, im3, im4 ],dtype=object)

            im5 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_msk2_val.npy')
            im6 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_msk2_val.npy')
            im7 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_msk2_val.npy')
            im8 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_msk2_val.npy')
            self.target_masks = np.array([im5, im6, im7, im8 ],dtype=object)

        elif(whichData=="train"):

            im = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_img2_trn.npy')
            im2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_img2_trn.npy')
            im3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_img2_trn.npy')
            im4 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_img2_trn.npy')

            self.input_images = np.array([im, im2, im3, im4 ],dtype=object)

            im5 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_msk2_trn.npy')
            im6 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_msk2_trn.npy')
            im7 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_msk2_trn.npy')
            im8 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_msk2_trn.npy')
            self.target_masks = np.array([im5, im6, im7, im8 ],dtype=object)

        elif(whichData=="predict"):

            files = os.listdir(imageDir)

            rho = np.zeros((len(files),shape,shape,1))
            mask_list = []
            
            count = 0

            for file in files:

                im = cv2.imread(imageDir + file, cv2.IMREAD_GRAYSCALE)
                im2 = np.reshape(im,(1024,1024,1))
            
                meanIm, stdIm = np.mean(im), np.std(im)
                normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)

                rows,cols = np.shape(im)
                shape = 1024
                    
                # loop over the image and put the image into a numpy ndarray                                                                                             
                # each image is a (1024x1024x3) numpy array, where 3 is the number of colour channels                                                                    
                rho[count,0:shape,0:shape,] = normalize(im2[0:shape,0:shape,])
                count += 1

                z1 = np.zeros((1024,1024))                      
                mask_list.append(z1)

            mask_array = np.asarray(mask_list)
            self.input_images = rho         
            self.target_masks = mask_array

        self.transform = transform      

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        i,x,y = idx

        image = self.input_images[i][x:(x+1024),y:(y+1024)] 
        mask = self.target_masks[i][x:(x+1024),y:(y+1024)] 

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return [image, mask]

