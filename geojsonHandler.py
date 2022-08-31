import glob
import argparse
import cv2
import os
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from matplotlib import pyplot
import geojson 
import time
import math
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, shape

def writeToGeoJSON(masterList, filename):
    number=0
    features = []
    for poly in masterList: 
        if poly.type == 'Polygon':
            features.append(Feature(geometry=Polygon([(x,y) for x,y in poly.exterior.coords]), properties={"id": str(number)}))
        if poly.type == 'MultiPolygon':            
            mycoordslist = [list(x.exterior.coords) for x in poly.geoms]
            ll=[x for xs in mycoordslist for x in xs]
            features.append(Feature(geometry=Polygon(ll), properties={"id": str(number)}))
        number+=1
    feature_collection = FeatureCollection(features)        
    ## I add another key to the FeatureCollection dictionary where it can look up max ID, so it knows which ID to work on when adding elements
    feature_collection["maxID"]=number
    with open(filename,"w") as outfile:
        dump(feature_collection, outfile) 
    outfile.close()


def addToGeoJSON(filename,polyToAdd):
    with open(filename) as f:
        gj = geojson.load(f)
    f.close()
    number = gj["maxID"]
    if polyToAdd.type == 'Polygon':
        gj['features'].append(Feature(geometry=Polygon([(x,y) for x,y in polyToAdd.exterior.coords]), properties={"id": str(number)}))
    if polyToAdd.type == 'MultiPolygon':
        mycoordslist = [list(x.exterior.coords) for x in polyToAdd.geoms]
        ll=[x for xs in mycoordslist for x in xs]
        gj['features'].append(Feature(geometry=Polygon(ll), properties={"id": str(number)}))
    ##newname= filename.replace(".geojson","_ADDED.geojson")
    ## just overwrite original geoJSON file
    with open(filename,"w") as outfile:
        dump(gj, outfile)
    outfile.close()   

def readGeoJSON2list(filename):
    with open(filename) as f0:
        gj0 = geojson.load(f0)
    f0.close()
    polyList = geojson2polygon(gj0)
    return(polyList)

# little function for converting feature elements from geojson into shapely polygon objects    
def geojson2polygon(gj):
    pols=[]
    for i in range(len(gj['features'])):
        pols.append(shape(gj['features'][i]["geometry"]))
    return(pols)

