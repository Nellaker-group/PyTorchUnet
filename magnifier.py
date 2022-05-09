import cv2
import numpy as np
from PIL import Image

def magnify(image,umSource,umTarget,input512,inputChannels,ifMask):
    size=1024
    if(input512==1):
        size=512
    assert umSource+0.001 > umTarget
    ratio = umTarget / umSource
    newSize=int(size*(ratio))
    # to get uint8 that PIL works with
    newImage = image.astype(np.uint8)
    if(inputChannels==3 and not ifMask):
        im2 = Image.fromarray(newImage[0:newSize,0:newSize])
    else:
        im2 = Image.fromarray(newImage[0:newSize,0:newSize],"L")
    im3 = im2.resize((size,size))
    im4 = np.asarray(im3)
    # to get float32 dtype that pipeline is designed for
    im5 = np.float32(im4)
    im6 = im5
    #if(inputChannels==3 and not ifMask):
    #    im6 = cv2.cvtColor(im5/255.0, cv2.COLOR_RGB2BGR)
    return(im6)
    



