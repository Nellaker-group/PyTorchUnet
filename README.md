# PyTorchUnet

U-net implementation in PyTorch of Craig Glastonbury's adipocyte pipeline in Keras
https://github.com/GlastonburyC/Adipocyte-U-net

So far it is doing binary segmentation (adipocyte v. non-adipocyte). The training data is collages of images (grayscale, 1 byte per pixel) and corresponding masks.
The files are stored as .npy files, one for each cohort (GTEx, ENDOX, julius/MOBB and fatdiva).

The montages consist of this many 1024x1024 tiles:

TRAINING
fatdiva: (6144 // 1024) * (10240 // 1024) = 60 tiles
gtex: (76500 // 1024) * (17000 // 1024) = 1184 tiles
endox: (9216 // 1024) * (5120 // 1024) = 45 tiles
julius/MOBB: (15360 // 1024) * (18432 // 1024) = 270 tiles (NOTICE THIS DATASET IS NOT H&E and therefore visually very different)

1184 tiles + 270 tiles + 60 tiles + 45 tiles = 1559 tiles

VALIDATION
fatdiva: (2048 // 1024) * (4096 // 1024) = 8
gtex: (8500 // 1024) * (6800 // 1024) =  48
endox: (7168 // 1024) * (2048 // 1024) = 14
julius/MOBB: (3840 // 1024) * (1024 // 1024) = 3 

8 tiles + 48 tiles + 14 tiles + 3 tiles = 73 tiles

##################################

(04-05-2022)

I have added zoom functionality (using --zoomFile) scaling datasets to a reference dataset, example:
endox 0.2500 reference
gtex 0.4942
fatdiva 0.6400

For doing preditions one has to specify the dataset one wants to predict on (for example "gtex", "endox" or "fatdiva").
For training folders should have same names as datasets in zoom file.
Example:

all_annotations/
all_annotations/endox
all_annotations/endox/trn_imgs
all_annotations/endox/vld_imgs
all_annotations/endox/trn_msks
all_annotations/endox/vld_msks
all_annotations/gtex
all_annotations/gtex/trn_imgs
all_annotations/gtex/vld_imgs
all_annotations/gtex/trn_msks
all_annotations/gtex/vld_msks


###############################

(13-12-2024)

Example run:

python -u main.py --gpu 0 --mode train --seed 36232 --trainDir /gpfs3/well/lindgren/users/swf744/adipocyte/segmentation/all_annotations_JPGwithBootstrapV2_munichLeipzigHohenheimV2_noFatdivaV2_cleanAnno_withTest/train/ --valDir /gpfs3/well/lindgren/users/swf744/adipocyte/segmentation/all_annotations_JPGwithBootstrapV2_munichLeipzigHohenheimV2_noFatdivaV2_cleanAnno_withTest/train/ --imageDir 1 --epochs 200 --tiles 200 --augment 1 --optimiser 1 --gamma 0.5 --stepSize 30 --torchSeed 456546 --LR 0.0001 --frankenstein 1 --normFile weights/normSeqTiles_ep60t2022_02_22-125709g0.5s865au1op1st30sB0LR0.0001fr0.norm --512 0 --inputChannels 3 --zoomFile zoomFileMunichLeipzigHohenheim.txt

Command line options explained:

--gpu, help='which GPU to run on', type=str
--mode, help='train or predict', type=str
--tiles, help='how many tiles to use for training', type=int, default=200
--seed, help='seed to use', type=int
--trainDir, help='path of directory for training data', type=str
--valDir, help='path of directory for validation data', type=str
--preDir, help='path of directory for predictions', type=str
--imageDir, help='if training data is directory with images', type=int
--epochs, help='number of epochs', type=int
--gamma, help='number of epochs', type=float, default=0
--weights, help='path to weights', type=str
--augment, help='whether to augment training', type=int, default=0
--optimiser, help='which optimiser to use, (cyclicLR=0, stepLR=1)', type=int, default=0
--stepSize, help='which step size to use for stepLR optimiser (--optimiser 1'), type=int, default=0
--torchSeed, help='seed for PyTorch so can control initialization  of weights', type=int, default=0
--frankenstein, help='assembles tiles from 4 different parts from different tiles (works for montages and uniform sampling across datasets)\n 1=Cuts 4 random parts from tiles and merges them together\n 2=Cuts 4 corners from tiles from the same dataset and merges them together, type=int, default=0
--sizeBasedSamp, help='if sampling from datasets should depend on the size of the datasets (yes=1, no=0)', type=int, default=0
--LR, help='start learning rate', type=float
--inputChannels, help='number of input channels - only works for values != 1 with --imageDir 1', type=int, default=1
--outputChannels, help='number of output channels or classes to predict', type=int, default=2
--trainingChannelsMultiplier, help='multiplier for number of training channels in U-net', type=int, default=1
--normFile, help='file with mean and SD for normalisation (1st line mean, 2nd line SD)', type=str
--zoomFile, help='file with how many um one pixel is for the different datasets (optional)', type=str, default=""
--whichDataset, help='which Dataset are we doing predictions in', type=str, default=""
--512, help='image size is 512, cuts existing 1024x1024 tiles into 4 bits', type=int, default=0
--predThres, help='threshold for predictions and creating mask - default is 0.8', type=float, default=0.8
--dirtyPredict, help='a nasty dirty way of doing predictions on a test set disguised as validation', type=int, default=0
