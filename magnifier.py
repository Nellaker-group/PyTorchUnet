import cv2
import numpy as np
from PIL import Image

def magnifyOneTile(image,mask,umSource,umTarget,input512,inputChannels):
    size=1024
    if(input512==1):
        size=512
    assert umSource+0.001 > umTarget
    ratio = umTarget / umSource
    newSize=int(size*(ratio))
    # to get uint8 that PIL works with
    newImage = image.astype(np.uint8)
    newMask = mask.astype(np.uint8)
    indent=(0,0)
    if(size>newSize):
        indent = sampleTopLeft(size, newSize)
    assert (indent[0]+newSize) <= size, "cut for zooming goes out of tile"
    assert (indent[1]+newSize) <= size, "cut for zooming goes out of tile"
    if(inputChannels==3):
        im2 = Image.fromarray(newImage[indent[0]:(indent[0]+newSize),indent[1]:(indent[1]+newSize)])
        mask2 = Image.fromarray(newMask[indent[0]:(indent[0]+newSize),indent[1]:(indent[1]+newSize)],"L")
    else:
        im2 = Image.fromarray(newImage[indent[0]:(indent[0]+newSize),indent[1]:(indent[1]+newSize)],"L")
        mask2 = Image.fromarray(newMask[indent[0]:(indent[0]+newSize),indent[1]:(indent[1]+newSize)],"L")
    # make image larger
    im3 = im2.resize((size,size))
    mask3 = mask2.resize((size,size))
    im4 = np.asarray(im3)
    mask4 = np.asarray(mask3)
    # to get float32 dtype that pipeline is designed for
    im5 = np.float32(im4)
    mask5 = np.float32(mask4)
    return(im5, mask5)
    

# samples sub-tile from org-tile
def sampleTopLeft(orgShape, newShape):
    xTop = np.random.randint(0,orgShape)
    yTop = np.random.randint(0,orgShape)
    while((xTop+newShape) > orgShape or (yTop+newShape) > orgShape):
        xTop = np.random.randint(0, orgShape)
        yTop = np.random.randint(0, orgShape)
    return(xTop, yTop)
