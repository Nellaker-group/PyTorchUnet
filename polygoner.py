from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json

# expects mask to be 0 and 1
def draw_polygons_from_mask(mask,X,Y):
    w,h=np.shape(mask)    
    padded_mask=np.zeros((w+2,h+2),dtype="uint8")    
    padded_mask[1:(w+1),1:(h+1)] = mask           
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(padded_mask, 0.5, positive_orientation="low")  
    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        # Emil has added X and Y coordinates to get global WSI coordinates
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1 + X, row - 1 + Y)
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue
        polygons.append(poly)
        if poly.type == 'Polygon':
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
        elif poly.type == 'MultiPolygon':
            segmentation = np.array(poly.geoms).ravel().tolist()
        segmentations.append(segmentation)
    # checking that polygons are not contained in another polygon
    polygonsKeep = []
    segmentationsKeep = []
    for j in range(0, len(polygons)):                
        contained=False
        intersected=False
        for i in range(0, len(polygons)):
            if polygons[j].contains(polygons[i]) and i != j:                
                contained=True
            if polygons[j].intersects(polygons[i]) and i != j:
                intersected=True
        if contained and intersected:
            polygonsKeep.append(polygons[j])
            segmentationsKeep.append(polygons[j])
        elif not intersected and not contained:
            polygonsKeep.append(polygons[j])
            segmentationsKeep.append(polygons[j])
    return polygonsKeep, segmentationsKeep
