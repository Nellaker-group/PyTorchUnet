
import openslide as osl
from os.path import basename
from os.path import splitext, join
import os
from PIL import Image
import csv
import numpy as np
import pandas as pd
import os
from scipy import ndimage, misc
import scipy.misc
import re,glob
import cv2
import sys
import os
from PIL import Image

## index, 0 will give you the python filename being executed. Any index after that are the arguments passed.
fileName = sys.argv[1] 
boundsx = sys.argv[2] 
boundsy = sys.argv[3] 

sizesy = sys.argv[4] 
sizesx = sys.argv[5] 
targetDir = sys.argv[6] 
if512 = sys.argv[7] 
zoomFile = sys.argv[8] 
sourceDataset = sys.argv[9] 
shift = sys.argv[10]

shiftx =  int(shift.split(",")[0])
shifty =  int(shift.split(",")[1])

print(fileName)

slide = osl.OpenSlide(fileName)

## if not .scn file
if(int(sizesx)==0):
    (sizesx, sizesy) = slide.dimensions

shape=1024
if(int(if512)==1):
    shape=512

## how much it has to move tiles to the left and up from original (0,0) start pos
antishiftx = shape - shiftx
antishifty = shape - shifty

refData = ""
zoomDict = {}

if(zoomFile!=""):
    with open(zoomFile,"r") as zf:
        firstLine = 1
        for line in zf:
            if(firstLine==1):
                d0, z0, m0 = tuple(line.split(" "))
                m0 = m0.strip()
                assert m0 == "reference"
                firstLine=0
                refData = d0.lower()
                zoomDict[refData] = float(z0)
            else:
                d0, z0 = tuple(line.split(" "))
                zoomDict[d0.lower()] = float(z0)                
    zf.close()

if zoomFile!="":
    ## pixel to micrometer GTEX (0.4942) and ENDOX (0.2500)
    ratio = (zoomDict[refData]/zoomDict[sourceDataset.lower()])
    shape = int(shape * ratio)
    shiftx = int(shiftx * ratio) 
    shifty = int(shifty * ratio)
    antishiftx = int(antishiftx * ratio) 
    antishifty = int(antishifty * ratio)

print("two first values are shifts used for new tiling")
print("two other values are how much first top left corner tile is shifted by - has to be minus to get top left corner!")
print(shiftx)
print(shifty)
print(antishiftx)
print(antishifty)

for x in range(0, int(sizesx)+shiftx, shape):
    for y in range(0, int(sizesy)+shifty, shape):
        newx = x+int(boundsx)-antishiftx
        newy = y+int(boundsy)-antishifty
        printx = x-antishifty
        printy = y-antishifty
        im=slide.read_region((newx,newy),0,(shape,shape))                    
        if(zoomFile!="" and int(if512)==1):
            im=im.resize((512,512))
        elif(zoomFile!="" and int(if512)==0):
            im=im.resize((1024,1024))
        fileName2=os.path.basename(fileName)
        fileName2=fileName2.replace(" ","_")
        fileName2=fileName2.replace(".scn","_X"+str(printx)+"_Y"+str(printy)+".png")
        fileName2=fileName2.replace(".svs","_X"+str(printx)+"_Y"+str(printy)+".png")
        im.save(targetDir+"/"+fileName2)




