# PyTorchUnet

U-net implementation in PyTorch of Craig Glastonbury's adipocyte pipeline in Keras
https://github.com/GlastonburyC/Adipocyte-U-net

So far it is doing binary segmentation (adipocyte v. non-adipocyte). The training data is collages of images (grayscale, 1 byte per pixel) and corresponding masks.
The files are stored as .npy files, one for each cohort (GTEx, NDOG, julius/MOBB and exeter).
