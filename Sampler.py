import torch
import math
import numpy as np
from torch.utils.data.sampler import Sampler



class RandomSamplerUniform(Sampler):
    """Samples tiles randomly from montages, with replacement.
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
        first = 1

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)

            rand_var = np.random.randint(0,4)

            if rand_var == 0:
                x0, y0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
            elif rand_var == 1:
                x0, y0 = np.random.randint(0, rangeH2), np.random.randint(0, rangeW2)
            elif rand_var == 2:
                x0, y0 = np.random.randint(0, rangeH3), np.random.randint(0, rangeW3)
            elif rand_var == 3:
                x0, y0 = np.random.randint(0, rangeH4), np.random.randint(0, rangeW4)
                
            listie.append((rand_var,x0,y0,first))
            first = 0

            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class RandomSamplerDatasetSize(Sampler):
    """Samples tiles randomly from montages, but taking size of each montage into account, with replacement.
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

        # to make sure we do not get np.random.randint(0, 0) which gives an error
        rangeH = max(H - wdw_H,1)
        rangeW = max(W - wdw_W,1)
        rangeH2 = max(H2 - wdw_H,1)
        rangeW2 = max(W2 - wdw_W,1)
        rangeH3 = max(H3 - wdw_H,1)
        rangeW3 = max(W3 - wdw_W,1)
        rangeH4 = max(H4 - wdw_H,1)
        rangeW4 = max(W4 - wdw_W,1)

        # to get how many tiles in each dataset
        tiles = math.floor(H//1024)*math.floor(W//1024)
        tiles2 = math.floor(H2//1024)*math.floor(W2//1024)
        tiles3 = math.floor(H3//1024)*math.floor(W3//1024)
        tiles4 = math.floor(H4//1024)*math.floor(W4//1024)

        tilesAll = tiles+tiles2+tiles3+tiles4

        listie=[]
        first = 1

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)

            rand_var = np.random.randint(0,tilesAll)
            whichDataset = -9

            if rand_var < tiles:
                x0, y0 = np.random.randint(0, rangeH), np.random.randint(0, rangeW)
                whichDataset = 0
            elif rand_var >= tiles and rand_var < (tiles+tiles2):
                x0, y0 = np.random.randint(0, rangeH2), np.random.randint(0, rangeW2)
                whichDataset = 1
            elif rand_var >= (tiles+tiles2) and rand_var < (tiles+tiles2+tiles3):
                x0, y0 = np.random.randint(0, rangeH3), np.random.randint(0, rangeW3)
                whichDataset = 2
            elif rand_var >= (tiles+tiles2+tiles3) and rand_var < (tiles+tiles2+tiles3+tiles4):
                x0, y0 = np.random.randint(0, rangeH4), np.random.randint(0, rangeW4)
                whichDataset = 3
                
            assert whichDataset >= 0
            listie.append((whichDataset,x0,y0,first))
            first = 0

            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class SeqSamplerUniform(Sampler):
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

        # to make sure we do not get np.random.randint(0, 0) which gives an error
        rangeH = max(H - wdw_H,1)
        rangeW = max(W - wdw_W,1)
        rangeH2 = max(H2 - wdw_H,1)
        rangeW2 = max(W2 - wdw_W,1)
        rangeH3 = max(H3 - wdw_H,1)
        rangeW3 = max(W3 - wdw_W,1)
        rangeH4 = max(H4 - wdw_H,1)
        rangeW4 = max(W4 - wdw_W,1)

        first = 1 
        listie=[]

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)
            rand_var = np.random.randint(0,4)
            index=-1

            if rand_var == 0:
                x0, y0 = np.random.choice(list(range(0, (rangeH+1), 1024))), np.random.choice(list(range(0, (rangeW+1), 1024)))
                index=0
            elif rand_var == 1:
                x0, y0 = np.random.choice(list(range(0, (rangeH2+1), 1024))), np.random.choice(list(range(0, (rangeW2+1), 1024)))
                index=1
            elif rand_var == 2:
                x0, y0 = np.random.choice(list(range(0, (rangeH3+1), 1024))), np.random.choice(list(range(0, (rangeW3+1), 1024)))
                index=2
            elif rand_var == 3:
                x0, y0 = np.random.choice(list(range(0, (rangeH4+1), 1024))), np.random.choice(list(range(0, (rangeW4+1), 1024)))
                index=3
                
            listie.append((index,x0,y0,first))
            first = 0
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class SeqSamplerDatasetSize(Sampler):
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
        first = 1

        for sample_idx in range(self.sample_size):

            x0, y0 = (0,0)
            rand_var = np.random.randint(0,100)
            index=-1

            # I am using the number of tiles in each montage (pre calculated) - to sample according to the size of each dataset
            if rand_var < 4:
                x0, y0 = rd.choice(list(range(0, (rangeH+1), 1024))), rd.choice(list(range(0, (rangeW+1), 1024)))
                index=0
            elif rand_var >= 4 and rand_var < 9:
                x0, y0 = rd.choice(list(range(0, (rangeH2+1), 1024))), rd.choice(list(range(0, (rangeW2+1), 1024)))
                index=1
            elif rand_var >= 9 and rand_var < 85:
                x0, y0 = rd.choice(list(range(0, (rangeH3+1), 1024))), rd.choice(list(range(0, (rangeW3+1), 1024)))
                index=2
            elif rand_var >= 85:                
                x0, y0 = rd.choice(list(range(0, (rangeH4+1), 1024))), rd.choice(list(range(0, (rangeW4+1), 1024)))
                index=3
                
            listie.append((index,x0,y0,first))
            first = 0
                        
            self.counter += 1

        return(iter(listie))

    def __len__(self):
        return len(self.data_source)


class ValSampler(Sampler):
    """Samples elements randomly, with replacement, but uniformly across datasets.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # so this has to have the data

    def __init__(self, data_source, shape, counter):

        self.data_source = data_source
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

        tiles = math.floor(H // wdw_H) * math.floor(W // wdw_H)
        tiles2 = math.floor(H2 // wdw_H) * math.floor(W2 // wdw_H)
        tiles3 = math.floor(H3 // wdw_H) * math.floor(W3 // wdw_H)
        tiles4 = math.floor(H4 // wdw_H) * math.floor(W4 // wdw_H)

        # to make sure we do not get np.random.randint(0, 0) which gives an error
        rangeH = max(H - wdw_H+1,1)
        rangeW = max(W - wdw_W+1,1)
        rangeH2 = max(H2 - wdw_H+1,1)
        rangeW2 = max(W2 - wdw_W+1,1)
        rangeH3 = max(H3 - wdw_H+1,1)
        rangeW3 = max(W3 - wdw_W+1,1)
        rangeH4 = max(H4 - wdw_H+1,1)
        rangeW4 = max(W4 - wdw_W+1,1)

        first = 1 
        listie=[]

        count = 0
        count2 = 0
        count3 = 0
        count4 = 0

        for x in range(0, rangeH, 1024):
            for y in range(0, rangeW, 1024):
                listie.append((0,x,y,first))
                first = 0
                count += 1

        for x in range(0, rangeH2, 1024):
            for y in range(0, rangeW2, 1024):
                listie.append((1,x,y,first))
                count2 += 1
                
        for x in range(0, rangeH3, 1024):
            for y in range(0, rangeW3, 1024):
                listie.append((2,x,y,first))
                count3 += 1
                
        for x in range(0, rangeH4, 1024):
            for y in range(0, rangeW4, 1024):
                listie.append((3,x,y,first))
                count4 += 1
                
        assert tiles == count
        assert tiles2 == count2
        assert tiles3 == count3
        assert tiles4 == count4
        
        return(iter(listie))

    def __len__(self):
        return len(self.data_source)

