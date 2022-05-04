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
