import torch
import numpy as np
import random
from torch.utils.data.sampler import Sampler



class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, sample_size, shape, val, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
        self.val = val
        self.counter = counter

    # this has to provide the iterator , the iterator will use the __getitem__ method

    def __iter__(self):        

        imgs = self.data_source.input_images[0]
        imgs2 = self.data_source.input_images[1]
        imgs3 = self.data_source.input_images[2]
        imgs4 = self.data_source.input_images[3]
        
        H, W = imgs.shape
        H2, W2 = imgs2.shape
        H3, W3 = imgs3.shape
        H4, W4 = imgs4.shape

        wdw_H, wdw_W = (1024,1024)
        _mean, _std = np.mean(imgs), np.std(imgs)
        _mean2, _std2 = np.mean(imgs2), np.std(imgs2)
        _mean3, _std3 = np.mean(imgs3), np.std(imgs3)
        _mean4, _std4 = np.mean(imgs4), np.std(imgs4)
        normalize = lambda x: (x - _mean) / (_std + 1e-10)
        normalize2 = lambda x: (x - _mean2) / (_std2 + 1e-10)
        normalize3 = lambda x: (x - _mean3) / (_std3 + 1e-10)
        normalize4 = lambda x: (x - _mean4) / (_std4 + 1e-10)

        listie=[]

        for sample_idx in range(self.sample_size):

            y0, x0 = (0,0)
            y1, x1 = (0,0)
            rand_var = np.random.randint(0,4)
            if rand_var == 0:
                y0, x0 = np.random.randint(0, H - wdw_H), np.random.randint(0, W - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
            if rand_var == 1:
                if self.val == True:
                    y0, x0 =  np.random.randint(0, H2 - wdw_H), 0
                else:
                    y0, x0 = np.random.randint(0, H2 - wdw_H), np.random.randint(0, W2 - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
            if rand_var == 2:
                if self.val == True:
                    y0, x0 = np.random.randint(0, H3 - wdw_H), np.random.randint(0, W3 - wdw_W)
                else:
                    y0, x0 = np.random.randint(0, H3 - wdw_H), np.random.randint(0, W3 - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
            if rand_var == 3:
                y0, x0 = np.random.randint(0, H4 - wdw_H), np.random.randint(0, W4 - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
                
            listie.append((rand_var,y0,x0))
            
        if not self.val:
            self.counter += 1
            print("My counter:")
            print(self.counter)


        return(iter(listie))

    def __len__(self):
        return len(self.data_source)
