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
import numpy as np
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from Dataset import GetDataMontage, GetDataFolder
from loss import calc_loss, calc_loss_val

def print_metrics(metrics, epoch_samples, phase, f, lr):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:f}".format(k, metrics[k] / epoch_samples))
    # to print learning rate, which we get using scheduler.get_last_lr()
    outputs.append("LR: {:e}".format(lr[0]))        
    print("{}: {}".format(phase, ", ".join(outputs)))
    print("{}: {}".format(phase, ", ".join(outputs)),file=f)

def print_metrics_val(metrics, epoch_samples_dict, phase, f, lr):
    outputs = []
    count = 0
    
    for k in metrics.keys():
        outputs.append("{}: {:f}".format(k, metrics[k] / epoch_samples_dict[k.split("_")[0]]))
        count += 1
        if count == 3:
            outputs.append("LR: {:e}".format(lr[0]))        
            print("{}: {}".format(phase, ", ".join(outputs)))
            print("{}: {}".format(phase, ", ".join(outputs)),file=f)
            count = 0
            outputs = []
        
# dumps first tile from training with its predicted mask and ground truth mask - uses pred threshold from predctions
def dump_predictions(labels,outputs,epoch,preName,phase,pred_threshold,shape=1024):
    labTmp=labels[0].detach().cpu().numpy()
    outTmp=torch.sigmoid(outputs)
    outTmp=outTmp[0].detach().cpu().numpy()
    outTmp[outTmp > pred_threshold] = 255
    outTmp[outTmp <= pred_threshold] = 0
    if phase == 'train':
        plt.imsave("crops"+preName+"/trainLabels_epoch"+str(epoch)+".png",  labTmp[0,0:shape,0:shape])
        plt.imsave("crops"+preName+"/trainPredicted_epoch"+str(epoch)+".png", outTmp[0,0:shape,0:shape])
    else:
        plt.imsave("crops"+preName+"/valLabels_epoch"+str(epoch)+".png",  labTmp[0,0:shape,0:shape])
        plt.imsave("crops"+preName+"/valPredicted_epoch"+str(epoch)+".png", outTmp[0,0:shape,0:shape])
        
def train_model(model, dataloaders, device, optimizer, scheduler, f, preName, whichOptim, pred_threshold, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    epoch_loss = 1e10
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        writePred = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            writePred=0
            if phase == 'train':
                # moved scheduler step to after optimiszer step according to pytorch documentation
                # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            metrics = defaultdict(float)
            epoch_samples = 0
            # for keeping track of number of samples from the different datasets (munich, gtex, ...) per epoch 
            epoch_samples_dict = {}
            # now also gets name of dataset from DataLoader or __getitem__ method for the Dataset class
            for inputs, labels, datasetNames in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # calculate loss per dataset for the validation - therefore different function
                    if phase == "val":
                        loss = calc_loss_val(outputs, labels, metrics, datasetNames, phase)
                    else:
                        loss = calc_loss(outputs, labels, metrics)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # if cyclicLR we also have to call scheduler here, to increase LR
                        if whichOptim == 0:
                            scheduler.step()
                # statistics - or number of samples run through in this epoch
                epoch_samples += inputs.size(0)
                # this is for counting how many samples from this epoch is from a specific dataset
                for datasetName in datasetNames:
                    if phase == "val":
                        try:
                            epoch_samples_dict[datasetName] += 1
                        except KeyError:
                            epoch_samples_dict[datasetName] = 1
                # writing training predictions
                if(writePred==0):
                    dump_predictions(labels,outputs,epoch,preName,phase,pred_threshold)
                    writePred=1                                    
            # this is for getting the total number of samples per epoch for reporting the total loss for validation
            epoch_samples_dict["TOTAL"] = epoch_samples
            if phase == "train":
                # this just prints loss, bce and dice for the whole dataset
                print_metrics(metrics, epoch_samples, phase, f, scheduler.get_last_lr())
                # the train loss is just called "loss"
                epoch_loss = metrics['loss'] / epoch_samples
            else:
                # this prints loss, bce and dice per dataset and for the whole dataset
                print_metrics_val(metrics, epoch_samples_dict, phase, f, scheduler.get_last_lr())
                # because the val loss is called TOTAL_loss
                epoch_loss = metrics['TOTAL_loss'] / epoch_samples            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        # moved this to here in accordance with HAPPY pipeline - but only for the exponential LR
        if whichOptim == 1:
            scheduler.step()
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
