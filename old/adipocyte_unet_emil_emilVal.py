# Unet implementation based on https://github.com/jocicmarko/ultrasound-nerve-segmentation
import numpy as np
np.random.seed(865)

from keras.models import Model
from keras.layers import (Input, merge, Conv2D, MaxPooling2D, 
                          UpSampling2D, Dropout, concatenate,
                          Conv2DTranspose, Lambda, Reshape, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from scipy.misc import imsave
from os import path, makedirs
import argparse
import keras.backend as K
import logging
import pickle
import tifffile as tiff
import os
import sys
sys.path.append('/well/lindgren/users/swf744/adipocyte/segmentation/')
from src.utils.runtime import funcname, gpu_selection
from src.utils.model import (dice_coef, dice_coef_loss, KerasHistoryPlotCallback, 
                             KerasSimpleLoggerCallback, jaccard_coef, jaccard_coef_int, 
                             weighted_bce_dice_loss, weighted_dice_loss, 
                             weighted_bce_loss, weighted_dice_coeff)
from src.utils.data import random_transforms
from src.utils.isbi_utils import isbi_get_data_montage

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.callbacks import CSVLogger
from keras.losses import binary_crossentropy
from src.utils.clr_callback import *
import random
from keras import regularizers
import glob
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math

sys.path.append('.')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# emil's loss
def bce_dice_lossV2(y_true, y_pred):
    return 0.5*binary_crossentropy(y_true, y_pred) + 0.5*dice_loss(y_true, y_pred)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
    

class UNet():

    def __init__(self, checkpoint_name):

        self.config = {
            'data_path': 'data',
            'input_shape': (1024, 1024),
            'output_shape': (1024, 1024),
            'transform_train': True,
            'batch_size': 2,            
            'nb_epoch': 60,
        }
        self.current_epoch_val = -1
        self.current_epoch_train = -1

        self.val_index = 0
        self.train_index = 0

        self.val_size = 72
        self.train_size = 200

        self.checkpoint_name = checkpoint_name
        self.net = None
        self.imgs_trn = None
        self.msks_trn = None
        self.imgs_val = None
        self.msks_val = None
        self.imgs_trn2 = None
        self.msks_trn2 = None
        self.imgs_val2 = None
        self.msks_val2 = None

        self.imgs_trn3 = None
        self.msks_trn3 = None
        self.imgs_val3 = None
        self.msks_val3 = None

        self.imgs_trn4 = None
        self.msks_trn4 = None
        self.imgs_val4 = None
        self.msks_val4 = None

        return

    @property
    def checkpoint_path(self):
        return '/well/lindgren/users/swf744/adipocyte/segmentation/checkpoints/%s_%d_dilation' % (self.checkpoint_name, self.config['input_shape'][0])

    def load_data(self):

        self.imgs_trn = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_img2_trn.npy')
        self.msks_trn = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_msk2_trn.npy')

        self.imgs_val = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_img2_val.npy')
        self.msks_val = np.load('/well/lindgren/craig/isbi-2012/gtex_montage/gtex_montage_msk2_val.npy')

        self.imgs_trn2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_img2_trn.npy')
        self.msks_trn2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_msk2_trn.npy')

        self.imgs_val2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_img2_val.npy')
        self.msks_val2 = np.load('/well/lindgren/craig/isbi-2012/julius_montage/julius_montage_msk2_val.npy')

        self.imgs_trn3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_img2_trn.npy')
        self.msks_trn3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_msk2_trn.npy')

        self.imgs_val3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_img2_val.npy')
        self.msks_val3 = np.load('/well/lindgren/craig/isbi-2012/NDOG_montage/NDOG_montage_msk2_val.npy')

        self.imgs_trn4 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_img2_trn.npy')
        self.msks_trn4 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_msk2_trn.npy')

        self.imgs_val4 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_img2_val.npy')
        self.msks_val4 = np.load('/well/lindgren/craig/isbi-2012/exeter_montage/exeter_montage_msk2_val.npy')

        _mean, _std = np.mean(self.imgs_trn), np.std(self.imgs_trn)
        _mean2, _std2 = np.mean(self.imgs_trn2), np.std(self.imgs_trn2)
        _mean3, _std3 = np.mean(self.imgs_trn3), np.std(self.imgs_trn3)
        _mean4, _std4 = np.mean(self.imgs_trn4), np.std(self.imgs_trn4)
        self.meanest = sum([_mean,_mean2,_mean3,_mean4]) / 4
        self.stdest = sum([_std,_std2,_std3,_std4]) / 4

        print("mean and std:")
        print(self.meanest)
        print(self.stdest)

        # all this for generating indexes for when sampling the val - so that the val is always the same each epoch
        H, W = self.imgs_val.shape
        H2, W2 = self.imgs_val2.shape
        H3, W3 = self.imgs_val3.shape
        H4, W4 = self.imgs_val4.shape

        wdw_H, wdw_W = (1024,1024)

        # to make sure we do not get np.random.randint(0, 0) which gives an error
        rangeH = max(H - wdw_H+1,1)
        rangeW = max(W - wdw_W+1,1)
        rangeH2 = max(H2 - wdw_H+1,1)
        rangeW2 = max(W2 - wdw_W+1,1)
        rangeH3 = max(H3 - wdw_H+1,1)
        rangeW3 = max(W3 - wdw_W+1,1)
        rangeH4 = max(H4 - wdw_H+1,1)
        rangeW4 = max(W4 - wdw_W+1,1)

        count = 0
        count2 = 0
        count3 = 0
        count4 = 0

        self.listie=[]

        for x in range(0, rangeH, 1024):
            for y in range(0, rangeW, 1024):
                self.listie.append((x,y,0))
                count += 1

        for x in range(0, rangeH2, 1024):
            for y in range(0, rangeW2, 1024):
                self.listie.append((x,y,1))
                count2 += 1
                
        for x in range(0, rangeH3, 1024):
            for y in range(0, rangeW3, 1024):
                self.listie.append((x,y,2))
                count3 += 1
                
        for x in range(0, rangeH4, 1024):
            for y in range(0, rangeW4, 1024):
                self.listie.append((x,y,3))
                count4 += 1

        print("gtex")
        print(count)

        print("moob/julius")
        print(count2)

        print("ndog")
        print(count3)

        print("exeter")
        print(count4)

        print("len(listie):")
        print(len(self.listie))

    def compile(self, init_nb=44, lr=0.0001, loss=bce_dice_loss):
        K.set_image_dim_ordering('tf')
        x = inputs = Input(shape=self.config['input_shape'], dtype='float32')
        x = Reshape(self.config['input_shape'] + (1,))(x)

        down1 = Conv2D(init_nb, 3, activation='relu', padding='same')(x)
        down1 = Conv2D(init_nb,3, activation='relu', padding='same')(down1)
        down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
        down2 = Conv2D(init_nb*2,3, activation='relu', padding='same')(down1pool)
        down2 = Conv2D(init_nb*2,3, activation='relu', padding='same')(down2)
        down2pool = MaxPooling2D((2,2), strides=(2, 2))(down2)
        down3 = Conv2D(init_nb*4,3, activation='relu', padding='same')(down2pool)
        down3 = Conv2D(init_nb*4,3, activation='relu', padding='same')(down3)
        down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

        # stacked dilated convolution
        dilate1 = Conv2D(init_nb*8,3, activation='relu', padding='same', dilation_rate=1)(down3pool)
        dilate2 = Conv2D(init_nb*8,3, activation='relu', padding='same', dilation_rate=2)(dilate1)
        dilate3 = Conv2D(init_nb*8,3, activation='relu', padding='same', dilation_rate=4)(dilate2)
        dilate4 = Conv2D(init_nb*8,3, activation='relu', padding='same', dilation_rate=8)(dilate3)
        dilate5 = Conv2D(init_nb*8,3, activation='relu', padding='same', dilation_rate=16)(dilate4)
        dilate6 = Conv2D(init_nb*8,3, activation='relu', padding='same', dilation_rate=32)(dilate5)

        dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])        
        up3 = UpSampling2D((2, 2))(dilate_all_added)
        up3 = Conv2D(init_nb*4,3, activation='relu', padding='same')(up3)
        up3 = concatenate([down3, up3])
        up3 = Conv2D(init_nb*4,3, activation='relu', padding='same')(up3)
        up3 = Conv2D(init_nb*4,3, activation='relu', padding='same')(up3)

        up2 = UpSampling2D((2, 2))(up3)
        up2 = Conv2D(init_nb*2,3, activation='relu', padding='same')(up2)
        up2 = concatenate([down2, up2])
        up2 = Conv2D(init_nb*2,3, activation='relu', padding='same')(up2)
        up2 = Conv2D(init_nb*2,3, activation='relu', padding='same')(up2)

        up1 = UpSampling2D((2, 2))(up2)
        up1 = Conv2D(init_nb,3, activation='relu', padding='same')(up1)
        up1 = concatenate([down1, up1])
        up1 = Conv2D(init_nb,3, activation='relu', padding='same')(up1)
        up1 = Conv2D(init_nb,3, activation='relu', padding='same')(up1)

        x = Conv2D(2, 1, activation='softmax')(up1)
        x = Lambda(lambda x: x[:, :, :, 1], output_shape=self.config['output_shape'])(x)
        self.net = Model(inputs=inputs, outputs=x)

        # Emil added to be able to print learning rate
        RMSoptimiser = RMSprop()
        lr_metric = get_lr_metric(RMSoptimiser)

        self.net.compile(optimizer=RMSoptimiser, loss=loss, metrics=[dice_coef,lr_metric])
        return

    def train(self):

        logger = logging.getLogger(funcname())

        gen_trn = self.batch_gen_trn(imgs=self.imgs_trn, 
                                     imgs2=self.imgs_trn2, 
                                     imgs3=self.imgs_trn3, 
                                     imgs4=self.imgs_trn4,
                                     msks=self.msks_trn,
                                     msks2=self.msks_trn2, 
                                     msks3=self.msks_trn3, 
                                     msks4=self.msks_trn4, 
                                     batch_size=self.config['batch_size'], 
                                                            transform=self.config['transform_train'],val=False)
        gen_val = self.batch_gen_val(imgs=self.imgs_val,
                                     imgs2=self.imgs_val2, 
                                     imgs3=self.imgs_val3, 
                                     imgs4=self.imgs_val4,
                                     msks=self.msks_val, 
                                     msks2=self.msks_val2, 
                                     msks3=self.msks_val3, 
                                     msks4=self.msks_val4, 
                                     batch_size=self.config['batch_size'], 
                                                            transform=self.config['transform_train'])

        csv_logger = CSVLogger('training.log')
        clr_triangular = CyclicLR(mode='triangular')
        clr_triangular._reset(new_base_lr=0.00001, new_max_lr=0.0005)
        cb = [clr_triangular, 
              EarlyStopping(monitor='val_loss', 
                            min_delta=1e-3, 
                            patience=300, 
                            verbose=1, 
                            mode='min'
                           ),
              ModelCheckpoint(self.checkpoint_path + '/weights_loss_val.weights',
              monitor='val_loss', 
              save_best_only=True, 
              verbose=1
              ),
              ModelCheckpoint(self.checkpoint_path + '/weights_loss_trn.weights',
              monitor='loss', 
              save_best_only=True, verbose=1
              ),
              csv_logger]
        print("self.checkpoint_path:")
        print(self.checkpoint_path)
        logger.info('Training for %d epochs.' % self.config['nb_epoch'])

        # emil - for getting sames tiles as in PyTorch
        np.random.seed(865)
        self.net.fit_generator(generator=gen_trn, steps_per_epoch=(self.train_size/self.config['batch_size']), epochs=self.config['nb_epoch'],
                               validation_data=gen_val, validation_steps=self.val_size, verbose=1, callbacks=cb)
                               # validation_data=gen_val, validation_steps=36, verbose=1, callbacks=cb)

        return

    def batch_gen_trn(self, imgs, imgs2, imgs3, 
                      imgs4, msks, msks2, msks3, 
                      msks4, batch_size, transform=True,
                      val=False):

        H, W = imgs.shape
        H2, W2 = imgs2.shape
        H3, W3 = imgs3.shape
        H4, W4 = imgs4.shape
        wdw_H, wdw_W = self.config['input_shape']
        normalize = lambda x: (x - self.meanest) / (self.stdest + 1e-10)

        while True:

            img_batch = np.zeros((batch_size,) + self.config['input_shape'], dtype=imgs.dtype)
            msk_batch = np.zeros((batch_size,) + self.config['output_shape'], dtype=msks.dtype)

            for batch_idx in range(batch_size):
                
                rand_var = np.random.randint(0,4)
                y0, x0 = (0,0)
                # emil changed
                # rand_var = random.random()
                #if rand_var < 0.25:
                if rand_var == 2:
                    #y0, x0 = np.random.randint(0, H - wdw_H), np.random.randint(0, W - wdw_W)
                    y0, x0 = np.random.randint(0, max(1,H - wdw_H)), np.random.randint(0, max(1,W - wdw_W))
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks[y0:y1, x0:x1]
                if rand_var == 3:
                #if rand_var >= 0.25 and rand_var < 0.50:
                    if val ==True:
                        y0, x0 =  np.random.randint(0, H2 - wdw_H), 0
                    else:
                        #y0, x0 = np.random.randint(0, H2 - wdw_H), np.random.randint(0, W2 - wdw_W)
                        y0, x0 = np.random.randint(0, max(1,H2 - wdw_H)), np.random.randint(0, max(1,W2 - wdw_W))
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs2[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks2[y0:y1, x0:x1]
                #if rand_var >= 0.50 and rand_var <= 0.75:
                if rand_var == 0:
                    if val == True:
                        y0, x0 = np.random.randint(0, H3 - wdw_H), np.random.randint(0, W3 - wdw_W)
                    else:
                        #y0, x0 = np.random.randint(0, H3 - wdw_H), np.random.randint(0, W3 - wdw_W)
                        y0, x0 = np.random.randint(0, max(1,H3 - wdw_H)), np.random.randint(0, max(1,W3 - wdw_W))
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs3[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks3[y0:y1, x0:x1]
                #if rand_var > 0.75:
                if rand_var == 1:
                    #y0, x0 = np.random.randint(0, H4 - wdw_H), np.random.randint(0, W4 - wdw_W)
                    y0, x0 = np.random.randint(0, max(1,H4 - wdw_H)), np.random.randint(0, max(1,W4 - wdw_W))
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs4[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks4[y0:y1, x0:x1]

                self.train_index += 1
                # because 0 indexed and
                if self.train_index == 1 and self.current_epoch_train > -1:
                    plt.imsave("valTrainDumpsV2/train_"+str(self.train_index)+"_epochs"+str(self.current_epoch_train)+"_randVar"+str(rand_var)+"_x"+str(x0)+"_y"+str(y0)+".png", img_batch[batch_idx])
                if self.train_index == 10 and self.current_epoch_train == -1:
                    self.train_index = 0
                    self.current_epoch_train += 1
                    # emil set seed so we get the same sampled values as pytorch
                    np.random.seed(865)
                elif self.train_index == self.train_size:
                    self.train_index = 0
                    self.current_epoch_train += 1
            
                img_batch = normalize(img_batch)
            yield img_batch, msk_batch
            

    def predict(self, imgs):
        imgs = (imgs - np.mean(imgs)) / (np.std(imgs) + 1e-10)
        return self.net.predict(imgs).round()


    def batch_gen_val(self, imgs, imgs2, imgs3, 
                      imgs4, msks, msks2, msks3, 
                      msks4, batch_size, transform=True):

        H, W = imgs.shape
        H2, W2 = imgs2.shape
        H3, W3 = imgs3.shape
        H4, W4 = imgs4.shape
        wdw_H, wdw_W = self.config['input_shape']

        normalize = lambda x: (x - self.meanest) / (self.stdest + 1e-10)

        # how many tiles wide
        tilesH = math.floor(H // wdw_H)
        tiles2H = math.floor(H2 // wdw_H)
        tiles3H = math.floor(H3 // wdw_H)
        tiles4H = math.floor(H4 // wdw_H)

        tilesW = math.floor(W // wdw_H)
        tiles2W = math.floor(W2 // wdw_H)
        tiles3W = math.floor(W3 // wdw_H)
        tiles4W = math.floor(W4 // wdw_H)

        img_batch = np.zeros((batch_size,) + self.config['input_shape'], dtype=imgs.dtype)
        msk_batch = np.zeros((batch_size,) + self.config['output_shape'], dtype=msks.dtype)

        while True:
        
            for batch_idx in range(batch_size):
            
                x,y,i = self.listie[self.val_index]

                if i == 0:
                    img_batch[batch_idx] = normalize(imgs[x:(x+1024),y:(y+1024)])
                    msk_batch[batch_idx] = msks[x:(x+1024),y:(y+1024)]
                elif i == 1:
                    img_batch[batch_idx] = normalize(imgs2[x:(x+1024),y:(y+1024)])
                    msk_batch[batch_idx] = msks2[x:(x+1024),y:(y+1024)]
                elif i == 2:
                    img_batch[batch_idx] = normalize(imgs3[x:(x+1024),y:(y+1024)])
                    msk_batch[batch_idx] = msks3[x:(x+1024),y:(y+1024)]
                elif i == 3:
                    img_batch[batch_idx] = normalize(imgs4[x:(x+1024),y:(y+1024)])
                    msk_batch[batch_idx] = msks4[x:(x+1024),y:(y+1024)]

                self.val_index += 1
                # because 0 indexed and
                
                plt.imsave("valTrainDumpsV2/val_"+str(self.val_index)+"_epochs"+str(self.current_epoch_val)+".png", img_batch[batch_idx])
                if self.val_index == 10 and self.current_epoch_val == -1:
                    self.val_index = 0
                    self.current_epoch_val += 1
                elif self.val_index == self.val_size:
                    self.val_index = 0
                    self.current_epoch_val += 1
            
                yield img_batch, msk_batch                

    def predict(self, imgs):
        imgs = (imgs - np.mean(imgs)) / (np.std(imgs) + 1e-10)
        return self.net.predict(imgs).round()


# from Tutorial.ipynb from https://github.com/GlastonburyC/Adipocyte-U-net/
def process_tiles(img_dir):
    tiles = glob.glob(img_dir +'*')
    samples = []
    print("HERE?")
    for i in tiles:
        s = cv2.imread(i,0)
        s = np.array(s,np.float32) /255
        print("image:")
        print(np.shape(s))
        #_mean, _std = np.mean(s), np.std(s)
        normalised_img = np.expand_dims((s - np.mean(s)) / np.std(s),0)
        #s = normalize(s)
        samples.append(normalised_img)
    samples=np.array(samples)
    return samples


def main():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(funcname())

    prs = argparse.ArgumentParser()
    prs.add_argument('--name', help='name used for checkpoints', default='unet', type=str)

    subprs = prs.add_subparsers(title='actions', description='Choose from:')
    subprs_trn = subprs.add_parser('train', help='Train the model.')
    subprs_sbt = subprs.add_parser('predict', help='Predict with the model.')
    subprs_trn.set_defaults(which='train')
    subprs_trn.add_argument('-w', '--weights', help='path to keras weights')
    subprs_sbt.set_defaults(which='predict')
    subprs_sbt.add_argument('-w', '--weights', help='path to weights', required=True)
    subprs_sbt.add_argument('-t', '--tiff', help='path to image')

    args = vars(prs.parse_args())
    assert args['which'] in ['train', 'predict']

    model = UNet(args['name'])

    if not path.exists(model.checkpoint_path):
        makedirs(model.checkpoint_path)

    def load_weights():
        if args['weights'] is not None:
            logger.info('Loading weights from %s.' % args['weights'])
            model.net.load_weights(args['weights'])

    if args['which'] == 'train':
        model.compile()
        load_weights()
        model.net.summary()
        model.load_data()
        model.train()
    elif args['which'] == 'predict':
        out_path = '%s/test-volume-masks.tif' % model.checkpoint_path
        model.config['input_shape'] = (1024, 1024)
        model.config['output_shape'] = (1024, 1024)
        model.compile()
        load_weights()
        model.net.summary()
        # imgs_sbt = tiff.imread(args['tiff'])

        imgs_sbt = process_tiles(args['tiff'])

        print("image:")
        print(np.shape(imgs_sbt))
        # emil
        # imgs_sbt = (0.299*imgs_sbt[:,:,0] + 0.587*imgs_sbt[:,:,1] + 0.114*imgs_sbt[:,:,2])
        # print("image:")
        # print(np.shape(imgs_sbt))
        

        msks_sbt = model.predict(imgs_sbt[0])
        logger.info('Writing predicted masks to %s' % out_path)
        tiff.imsave(out_path, msks_sbt)


if __name__ == "__main__":
    main()
