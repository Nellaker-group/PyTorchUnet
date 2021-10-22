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

    def __init__(self, data_source, sample_size, shape, counter):

        self.data_source = data_source
        self.sample_size = sample_size
        self.shape = shape
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

        # to make sure we do not get np.random.randint(0, 0) which gives an error
        rangeH = max(H - wdw_H,1)
        rangeW = max(W - wdw_W,1)
        rangeH2 = max(H2 - wdw_H,1)
        rangeW2 = max(W2 - wdw_W,1)
        rangeH3 = max(H3 - wdw_H,1)
        rangeW3 = max(W3 - wdw_W,1)
        rangeH4 = max(H4 - wdw_H,1)
        rangeW4 = max(W4 - wdw_W,1)

        listie=[]

        for sample_idx in range(self.sample_size):

            y0, x0 = (0,0)
            y1, x1 = (0,0)
            rand_var = np.random.randint(0,4)
            if rand_var == 0:
                y0, x0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
            if rand_var == 1:
                y0, x0 = np.random.randint(0, rangeH2), np.random.randint(0, rangeW2)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
            if rand_var == 2:
                y0, x0 = np.random.randint(0, rangeH3), np.random.randint(0, rangeW3)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
            if rand_var == 3:
                y0, x0 = np.random.randint(0, rangeH4), np.random.randint(0, rangeW4)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
                
            listie.append((rand_var,y0,x0))
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)
