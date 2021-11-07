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

            rand_var = np.random.randint(0,4)
            if rand_var == 0:
                y0, x0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
            elif rand_var == 1:
                y0, x0 = np.random.randint(0, rangeH2), np.random.randint(0, rangeW2)
            elif rand_var == 2:
                y0, x0 = np.random.randint(0, rangeH3), np.random.randint(0, rangeW3)
            elif rand_var == 3:
                y0, x0 = np.random.randint(0, rangeH4), np.random.randint(0, rangeW4)
                
            listie.append((rand_var,y0,x0))
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)




class SeqSampler(Sampler):
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

        print("imgs:")
        print(np.shape(imgs))
        print(imgs.__class__)
    
        l = len(imgs)
        l2 = len(imgs2)
        l3 = len(imgs3)
        l4 = len(imgs4)

        listie=[]

        for sample_idx in range(self.sample_size):

            rand_var = np.random.randint(0,4)
            i = -1
            if rand_var == 0:
                i = np.random.randint(0, l)
            elif rand_var == 1:
                i = np.random.randint(0, l2)
            elif rand_var == 2:
                i = np.random.randint(0, l3)
            elif rand_var == 3:
                i = np.random.randint(0, l4)
                
            listie.append((rand_var,i))
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)

    def __len__(self):
        return len(self.data_source)





class SeqSamplerV2(Sampler):
    """Samples elements randomly, with replacement, takes size of the datasets into account when sampling.
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
            rand_var = np.random.randint(0,100)
            index=-1

            # I am using the number of tiles in each montage (pre calculated) - to sample according to the size of each dataset
            if rand_var < 4:
                y0, x0 = random.choice(list(range(0, (rangeH+1), 1024))), random.choice(list(range(0, (rangeW+1), 1024)))
                index=0
            elif rand_var >= 4 and rand_var < 9:
                y0, x0 = random.choice(list(range(0, (rangeH2+1), 1024))), random.choice(list(range(0, (rangeW2+1), 1024)))
                index=1
            elif rand_var >= 9 and rand_var < 85:
                y0, x0 = random.choice(list(range(0, (rangeH3+1), 1024))), random.choice(list(range(0, (rangeW3+1), 1024)))
                index=2
            elif rand_var >= 85:                
                y0, x0 = random.choice(list(range(0, (rangeH4+1), 1024))), random.choice(list(range(0, (rangeW4+1), 1024)))
                index=3
                
            listie.append((index,y0,x0))
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)




class SeqSamplerV2uniform(Sampler):
    """Samples elements randomly, with replacement, but uniformly across datasets.
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
            rand_var = np.random.randint(0,4)
            index=-1

            if rand_var == 0:
                y0, x0 = random.choice(list(range(0, (rangeH+1), 1024))), random.choice(list(range(0, (rangeW+1), 1024)))
                index=0
            elif rand_var == 1:
                y0, x0 = random.choice(list(range(0, (rangeH2+1), 1024))), random.choice(list(range(0, (rangeW2+1), 1024)))
                index=1
            elif rand_var == 2:
                y0, x0 = random.choice(list(range(0, (rangeH3+1), 1024))), random.choice(list(range(0, (rangeW3+1), 1024)))
                index=2
            elif rand_var == 3:
                y0, x0 = random.choice(list(range(0, (rangeH4+1), 1024))), random.choice(list(range(0, (rangeW4+1), 1024)))
                index=3
                
            listie.append((index,y0,x0))
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)

