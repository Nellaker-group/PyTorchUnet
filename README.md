# PyTorchUnet

U-net implementation in PyTorch of Craig Glastonbury's adipocyte pipeline in Keras
https://github.com/GlastonburyC/Adipocyte-U-net

It does training for a binary segmentation (adipocyte v. non-adipocyte) U-net model. The training data is tiles with corresponding masks.
The training tiles should preferably be stored as PNG files (as these images are not compressed and therefore do not change pixel values because of compression).
For training folders should have same names as datasets in zoom file.
Example:
```
folder/
folder/train/endox/image.png
folder/train/endox/mask_image.png
folder/val/endox/image.png
folder/val/endox/mask_image.png
```
The names of the datasets in the "train/" and "val/" folders MUST correpsond to the name in the zoomFile.

###############################

DO GUIDE ON HOW TO USE QuPath for doing annotations and SAVING them with GROOVY scripts

###############################

(13-12-2024)

Example run:
```
python -u main.py --gpu 0 --mode train --seed 36232 --trainDir /gpfs3/well/lindgren/users/swf744/adipocyte/segmentation/all_annotations_JPGwithBootstrapV2_munichLeipzigHohenheimV2_noFatdivaV2_cleanAnno_withTest/train/ --valDir /gpfs3/well/lindgren/users/swf744/adipocyte/segmentation/all_annotations_JPGwithBootstrapV2_munichLeipzigHohenheimV2_noFatdivaV2_cleanAnno_withTest/train/ --imageDir 1 --epochs 200 --tiles 200 --augment 1 --optimiser 1 --gamma 0.5 --stepSize 30 --torchSeed 456546 --LR 0.0001 --frankenstein 1 --normFile weights/normSeqTiles_ep60t2022_02_22-125709g0.5s865au1op1st30sB0LR0.0001fr0.norm --512 0 --inputChannels 3 --zoomFile zoomFileMunichLeipzigHohenheim.txt
```
Command line options explained:
```
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
```

# How to annotate

Open your microscope file with adipose tissue in QuPath

Click on the "Annotations" on the left > Click the button with three vertical dots (in lower right corner of the box) > Click "Add/Remove..." > Click "Add Class" > enter "Adipocyte" as Class Name and click OK
Click on the cog wheel icon ("Preferences") > In that tab click on Viewer (and scroll down) > Then deselect "Grid Spacing in um" and change "Grid spacing X" and "Grid spacing Y" to the right values (in my case 1024)
Then click on the grid icon ("Show grid", next to the cog wheel icon) to show the grid imposed on the whole slide image
In the "Annotations" tab click on the "Adipocyte" class then click on the wand icon

Fill in the adipocyte in a chosen tile holding SHIFT+CTRL and then filling in the adipocyte with left click
You can delete by holding ALT and then removing with left click
The Brush icon can also be click for a regular brush without any automation (holding SHIFT+CTRL or ALT works here as well)

When you are done with a tile make sure the drawn polygons in the Annotations tab have the class "Adipocyte"
If they do not select all those (you can select more holding CTRL or SHIFT and then clicking) and then select the "Adipocyte" then press the "Set class" button

Then click on "Automate" in the menu bar then select "Show script editor" Open a New Script
Use the following groovy script (change the variables "localPath" and "tileSize"):

```
import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()
// Define path where to store tiles on your local computer (if Windows you need "\\" instead of "\")
def localPath = ""
// Define size of tiles, right now the training only works with tiles of size 1024x1024
def tileSize = 1024

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(localPath, '', name)
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 1.0
// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Adipocyte', 1, ColorTools.WHITE)      // One has to set a class (Polygon is not a class, when import from geoJSON) - set to class 'Other' when not using annotations
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(1)     // Define export resolution
    .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(tileSize)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  
    .overlap(0)                
    .writeTiles(pathOutput)     // Write tiles to the specified directory
print 'Done!'
```
Then click "Run" in the menu bar of the Script Editor and select "Run"
You then get a folder in the path pointed to by the "localPath" variable with the tile and the corresponding mask, these can be used for training the U-net

