from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

# dice loss per sample - as this is used when reporting loss per dataset, 
# as we have to calculate per sample in a batch, so it can get recorded to the right dataset
def dice_loss_val(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=1).sum(dim=1)    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + smooth)))
    return loss.mean()


# like glasonbury does it
def calc_loss(preds, targets, metrics):

    preds = torch.sigmoid(preds)
    bce = F.binary_cross_entropy(preds, targets)    
    dice = dice_loss(preds, targets)
    loss = bce + dice 
    metrics['bce'] += bce.data.cpu().numpy() * targets.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * targets.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * targets.size(0)
    return loss


# loss per dataset
def calc_loss_val(preds, targets, metrics, datasetNames, phase):

    reportLoss = 0
    for index in range(len(preds)):

        pred = torch.sigmoid(preds[index])
        bce = F.binary_cross_entropy(pred, targets[index])
    
        dice = dice_loss_val(pred, targets[index])
        loss = bce + dice 

        # this keeps track of bce, dice and loss but per dataset using dicts, the datasetNames dict has the names of the datasets
        # for putting it in the metrics dict - so it can record loss per dataset
        metrics[datasetNames[index]+'_bce'] += bce.data.cpu().numpy() * targets[index].size(0)
        metrics[datasetNames[index]+'_dice'] += dice.data.cpu().numpy() * targets[index].size(0)
        metrics[datasetNames[index]+'_loss'] += loss.data.cpu().numpy() * targets[index].size(0)

        # this keeps track of the total bce, dice and loss
        metrics['TOTAL_bce'] += bce.data.cpu().numpy() * targets[index].size(0)
        metrics['TOTAL_dice'] += dice.data.cpu().numpy() * targets[index].size(0)
        metrics['TOTAL_loss'] += loss.data.cpu().numpy() * targets[index].size(0)

        # this is the total loss over the batch - and should be reported
        reportLoss += loss

    return reportLoss
