from collections import defaultdict
from loss import dice_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
#import model1
import simulationV2
import numpy as np
import cv2
import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

# I have used this as an inspiration
# https://github.com/usuyama/pytorch-unet

# load all the images into "rho" (ndarray which holds all the training images) (from each cohort)
# load all the masks into "mask_list" (list that holds all the masks) (from each cohort)

class GetData(Dataset):
    def __init__(self, whichData, imageDir, transform=None):

        # define the size of the tiles to be working on
        shape = 1024

        assert whichData in ['train', 'validation', 'predict']

        if(whichData=="validation"):

            # we read the .png image into python as a numpy ndarray using the cv2.imread function
            im = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_img2_val.npy')
            print("im shape:")
            print(np.shape(im))
            rows,cols = np.shape(im)
            meanIm, stdIm = np.mean(im), np.std(im)
            normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)
            # getting the number of tiles the image contains
            tiles = math.floor(rows // shape) * math.floor(cols // shape)

            im2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_img2_val.npy')
            print("im2 shape:")
            print(np.shape(im2))
            rows2,cols2 = np.shape(im2)
            meanIm2, stdIm2 = np.mean(im2), np.std(im2)
            normalize2 = lambda x: (x - meanIm2) / (stdIm2 + 1e-10)
            # getting the number of tiles the image contains
            tiles2 = math.floor(rows2 // shape) * math.floor(cols2 // shape)

            im3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_img2_val.npy')
            print("im3 shape:")
            print(np.shape(im3))
            rows3,cols3 = np.shape(im3)
            meanIm3, stdIm3 = np.mean(im3), np.std(im3)
            normalize3 = lambda x: (x - meanIm3) / (stdIm3 + 1e-10)
            # getting the number of tiles the image contains
            tiles3 = math.floor(rows3 // shape) * math.floor(cols3 // shape)

            im4 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_img2_val.npy')
            print("im4 shape:")
            print(np.shape(im4))
            rows4,cols4 = np.shape(im4)
            meanIm4, stdIm4 = np.mean(im4), np.std(im4)
            normalize4 = lambda x: (x - meanIm4) / (stdIm4 + 1e-10)
            # getting the number of tiles the image contains
            tiles4 = math.floor(rows4 // shape) * math.floor(cols4 // shape)

            count = 0
            rho = np.zeros(((tiles+tiles2+tiles3+tiles4),shape,shape))
            # loop over the image and put the image into a numpy ndarray
            # each image is a (1024x1024x3) numpy array, where 3 is the number of colour channels
            for x in range(0, rows, shape):
                for y in range(0, cols, shape):
                    print("x, y:")
                    print(x)
                    print(y)
                    print(np.shape(im[x:(x+shape),y:(y+shape),]))
                    # I AM HERE!
                    if((x+shape)>rows or (y+shape)>cols):
                        continue
                    # reading it in row by row, so start top left then move right, and then down and all over
                    rho[count,0:shape,0:shape,] = normalize(im[x:(x+shape),y:(y+shape),])
                    count += 1

            for x in range(0, rows2, shape):
                for y in range(0, cols2, shape):
                    print("x, y:")
                    print(x)
                    print(y)
                    if((x+shape)>rows2 or (y+shape)>cols2):
                        continue
                    rho[count,0:shape,0:shape,] = normalize2(im2[x:(x+shape),y:(y+shape),])
                    count += 1

            for x in range(0, rows3, shape):
                for y in range(0, cols3, shape):
                    if((x+shape)>rows3 or (y+shape)>cols3):
                        continue
                    rho[count,0:shape,0:shape,] = normalize3(im3[x:(x+shape),y:(y+shape),])
                    count += 1

            for x in range(0, rows4, shape):
                for y in range(0, cols4, shape):
                    if((x+shape)>rows4 or (y+shape)>cols4):
                        continue
                    rho[count,0:shape,0:shape,] = normalize4(im4[x:(x+shape),y:(y+shape),])
                    count += 1

            self.input_images = rho            

            mask_list = []                
            im5 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_msk2_val.npy')
            rows5,cols5 = np.shape(im5)
            tiles5 = math.floor(rows5 // shape) * math.floor(cols5 // shape)
            
            im6 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_msk2_val.npy')
            rows6,cols6 = np.shape(im6)
            tiles6 = math.floor(rows6 // shape) * math.floor(cols6 // shape)
            
            im7 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_msk2_val.npy')
            rows7,cols7 = np.shape(im7)
            tiles7 = math.floor(rows7 // shape) * math.floor(cols7 // shape)

            im8 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_msk2_val.npy')
            rows8,cols8 = np.shape(im8)
            tiles8 = math.floor(rows8 // shape) * math.floor(cols8 // shape)

            for x in range(0, rows5, shape):
                for y in range(0, cols5, shape): 
                    if((x+shape)>rows5 or (y+shape)>cols5):
                        continue
                    # we read the image in and do a sum across the depth of it (depth = 3)
                    # and since we only have 2 colours and I know which one is the class and which one is background
                    # I convert it to a (1025x1025x1) ndarray
                    tmp3 = im5[x:(x+shape),y:(y+shape)]                 
                    # this is in order to make it to a 2D ndarray
                    tmp4 = tmp3
                    # this is in order to get the same dimensions as in the example where they do an array of the 2D ndarrays
                    masks = np.asarray([
                        tmp4,
                        #np.zeros((shape,shape)),
                    ]).astype(np.float32)
                    # they then put these arrays into a list
                    mask_list.append(masks)
                        
            for x in range(0, rows6, shape):
                for y in range(0, cols6, shape):
                    if((x+shape)>rows6 or (y+shape)>cols6):
                        continue
                    tmp3 = im6[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)

            for x in range(0, rows7, shape):
                for y in range(0, cols7, shape):
                    if((x+shape)>rows7 or (y+shape)>cols7):
                        continue               
                    tmp3 = im7[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)

            for x in range(0, rows8, shape):
                for y in range(0, cols8, shape):
                    if((x+shape)>rows8 or (y+shape)>cols8):
                        continue               
                    tmp3 = im8[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)
                                        
            print("validation tiles:")
            print(tiles+tiles2+tiles3+tiles4)

            # and finally converst the list into an array
            mask_array = np.asarray(mask_list)
            self.target_masks = mask_array

        elif(whichData=="train"):

            im = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_img2_trn.npy')
            rows,cols = np.shape(im)
            meanIm, stdIm = np.mean(im), np.std(im)
            normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)
            # getting the number of tiles the image contains
            tiles = math.floor(rows // shape) * math.floor(cols // shape)
        
            im2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_img2_trn.npy')
            rows2,cols2 = np.shape(im2)
            meanIm2, stdIm2 = np.mean(im2), np.std(im2)
            normalize2 = lambda x: (x - meanIm2) / (stdIm2 + 1e-10)
            # getting the number of tiles the image contains
            tiles2 = math.floor(rows2 // shape) * math.floor(cols2 // shape)
    
            im3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_img2_trn.npy')
            rows3,cols3 = np.shape(im3)
            meanIm3, stdIm3 = np.mean(im3), np.std(im3)
            normalize3 = lambda x: (x - meanIm3) / (stdIm3 + 1e-10)
            # getting the number of tiles the image contains
            tiles3 = math.floor(rows3 // shape) * math.floor(cols3 // shape)
    
            im4 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_img2_trn.npy')
            rows4,cols4 = np.shape(im4)
            meanIm4, stdIm4 = np.mean(im4), np.std(im4)
            normalize4 = lambda x: (x - meanIm4) / (stdIm4 + 1e-10)
            # getting the number of tiles the image contains
            tiles4 = math.floor(rows4 // shape) * math.floor(cols4 // shape)
    
            count = 0
            rho = np.zeros(((tiles+tiles2+tiles3+tiles4),shape,shape))
            # loop over the image and put the image into a numpy ndarray
            # each image is a (1024x1024x3) numpy array, where 3 is the number of colour channels
            for x in range(0, rows, shape):
                for y in range(0, cols, shape):
                    if((x+shape)>rows or (y+shape)>cols):
                        continue
                    # reading it in row by row, so start top left then move right, and then down and all over
                    rho[count,0:shape,0:shape,] = normalize(im[x:(x+shape),y:(y+shape),])
                    count += 1

            for x in range(0, rows2, shape):
                for y in range(0, cols2, shape):
                    if((x+shape)>rows2 or (y+shape)>cols2):
                        continue
                    rho[count,0:shape,0:shape,] = normalize2(im2[x:(x+shape),y:(y+shape),])
                    count += 1

            for x in range(0, rows3, shape):
                for y in range(0, cols3, shape):
                    if((x+shape)>rows3 or (y+shape)>cols3):
                        continue
                    rho[count,0:shape,0:shape,] = normalize3(im3[x:(x+shape),y:(y+shape),])
                    count += 1

            for x in range(0, rows4, shape):
                for y in range(0, cols4, shape):
                    if((x+shape)>rows4 or (y+shape)>cols4):
                        continue
                    rho[count,0:shape,0:shape,] = normalize4(im4[x:(x+shape),y:(y+shape),])
                    count += 1

            self.input_images = rho            

            mask_list = []                
            im5 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_msk2_trn.npy')
            rows5,cols5 = np.shape(im5)
            tiles5 = math.floor(rows5 // shape) * math.floor(cols5 // shape)

            im6 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_msk2_trn.npy')
            rows6,cols6 = np.shape(im6)
            tiles6 = math.floor(rows6 // shape) * math.floor(cols6 // shape)
            
            im7 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_msk2_trn.npy')
            rows7,cols7 = np.shape(im7)
            tiles7 = math.floor(rows7 // shape) * math.floor(cols7 // shape)

            im8 = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_msk2_trn.npy')
            rows8,cols8 = np.shape(im8)
            tiles8 = math.floor(rows8 // shape) * math.floor(cols8 // shape)

            for x in range(0, rows5, shape):
                for y in range(0, cols5, shape):
                    if((x+shape)>rows5 or (y+shape)>cols5):
                        continue                 
                    tmp3 = im5[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)
                        
            for x in range(0, rows6, shape):
                for y in range(0, cols6, shape):                       
                    if((x+shape)>rows6 or (y+shape)>cols6):
                        continue                 
                    tmp3 = im6[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)

            for x in range(0, rows7, shape):
                for y in range(0, cols7, shape):
                    if((x+shape)>rows7 or (y+shape)>cols7):
                        continue                 
                    tmp3 = im7[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)

            for x in range(0, rows8, shape):
                for y in range(0, cols8, shape):
                    if((x+shape)>rows8 or (y+shape)>cols8):
                        continue                 
                    tmp3 = im8[x:(x+shape),y:(y+shape)]                 
                    tmp4 = tmp3
                    masks = np.asarray([
                        tmp4,
                    ]).astype(np.float32)
                    mask_list.append(masks)

            print("training tiles:")
            print(tiles+tiles2+tiles3+tiles4)

            mask_array = np.asarray(mask_list)
            self.target_masks = mask_array

        elif(whichData=="predict"):

            files = os.listdir(imageDir)

            rho = np.zeros((len(files),shape,shape,1))
            mask_list = []
            
            count = 0

            for file in files:

                im = cv2.imread(imageDir + file, cv2.IMREAD_GRAYSCALE)
                print("shape:")
                print(np.shape(im))
                print("file:")
                print(imageDir + file)
                im2 = np.reshape(im,(1024,1024,1))
            
                meanIm, stdIm = np.mean(im), np.std(im)
                normalize = lambda x: (x - meanIm) / (stdIm + 1e-10)

                print("shape:")
                print(np.shape(im))
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
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        return [image, mask]


def get_training():

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    train_set = GetData("train", transform=trans)
    val_set = GetData("validation", transform=trans)
    
    image_datasets = {
        'train': train_set, 'val': val_set
    }
    
    batch_size = 2
    
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    print("Len:")
    print(len(train_set))
    print(len(val_set))
    
    return dataloaders


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


def dilate(in_channels, out_channels, dilation):
    return nn.Conv2d(in_channels, out_channels, dilation)     
    

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()                
        self.dconv_down1 = double_conv(1, 44)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_down2 = double_conv(44, 44*2)
        self.dconv_down3 = double_conv(44*2, 44*4)       
        self.dilate1 = nn.Conv2d(44*4, 44*8, 3, dilation=1)     
        self.dilate2 = nn.Conv2d(44*8, 44*8, 3, dilation=2)     
        self.dilate3 = nn.Conv2d(44*8, 44*8, 3, dilation=4)     
        self.dilate4 = nn.Conv2d(44*8, 44*8, 3, dilation=8)     
        self.dilate5 = nn.Conv2d(44*8, 44*8, 3, dilation=16)     
        self.dilate6 = nn.Conv2d(44*8, 44*8, 3, dilation=32)     
        self.upsample = nn.Upsample(scale_factor=2)        
        self.dconv_up3 = double_conv(44*8, 44*4)
        self.dconv_up2 = double_conv(44*4+44*2, 44*2)
        self.dconv_up1 = double_conv(44*2+44, 44)        
        self.conv_last = nn.Conv2d(44, n_class, 1)
        
        
    def forward(self, x):        
        # to convert the Tensor to have the data type of floats
        x = x.float()
        #x = torch.reshape(x, (3, 1024, 1024))
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        x1 = self.dilate1(x)
        x2 = self.dilate2(x1)
        x3 = self.dilate3(x2)
        x4 = self.dilate4(x3)
        x5 = self.dilate5(x4)
        x6 = self.dilate6(x5)
        # I AM NOT SURE HOW TO DO THIS
        # ASK ABOUT https://github.com/GlastonburyC/Adipocyte-U-net/blob/master/src/models/adipocyte_unet.py
        # Line 138-145
        # x = add([x1, x2, x3, x4, x5, x6])        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)           
        x = self.dconv_up1(x)        
        out = self.conv_last(x)        
        return out

# check keras-like model summary using torchsummary
# from torchsummary import summary
# summary(model, input_size=(3, 224, 224))


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, dataloaders, device, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# freeze backbone layers
#for l in model.base_layers:
#    for param in l.parameters():
#        param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, imageDir, device):

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        #transforms.Normalize([0.5], [0.5])
    ])

    pred_set = GetData("predict", imageDir, transform=trans)
    
    batch_size = 2
    
    test_loader = DataLoader(pred_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()

    files = os.listdir(imageDir)


    for i in range(0,len(pred)):
        newFile = files[i].replace(".png","_mask.png")
        newFile = files[i].replace(".jpg","_mask.png")

        newPred = pred[i][0]
    
        newPred[newPred > 0.5] = 255
        newPred[newPred <= 0.5] = 0
        plt.imsave(imageDir+newFile, newPred)

    return pred


def select_gpu(whichGPU):

    if(whichGPU=="0"):
        print("here0")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif(whichGPU=="1"):
        print("here1")
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif(whichGPU=="2"):
        print("here2")
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    elif(whichGPU=="3"):
        print("here3")
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        
    return device


def main():

    ## index, 0 will give you the python filename being executed. Any index after that are the arguments passed.
    gpu= sys.argv[1]
    trainOrPredict= sys.argv[2]

    if(len(sys.argv)>2):
        imageDir= sys.argv[3]

    assert trainOrPredict in ['train', 'predict']


    device = select_gpu(gpu)
    print(device)
        
    num_class = 1
    model = UNet(n_class=num_class).to(device)
        
    #model.double()
    print(model)
    
    print("N parameters:")
    print(count_parameters(model))

    if(trainOrPredict == "train"):
        training_data = get_training()
        optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
        model = train_model(model, training_data, device, optimizer_ft, exp_lr_scheduler, num_epochs=60)
        torch.save(model.state_dict(),"/gpfs3/well/lindgren/users/swf744/git/pytorch-unet/weights/weights.dat")
    else:

        # load image
        model.load_state_dict(torch.load("/gpfs3/well/lindgren/users/swf744/git/pytorch-unet/weights/weights.dat"))
        model.eval()
        results = predict(model,imageDir,device)


if __name__ == "__main__":
    main()

