from skimage.measure import regionprops
import cv2
import numpy as np
from scipy import ndimage
import pandas as pd
import argparse
import os
import glob
from skimage.measure import regionprops
from matplotlib import pyplot
from geojson import Point, Feature, FeatureCollection, dump
import sys
from PIL import Image
import shapely

#######################################################################
## based on https://github.com/chrise96/image-to-coco-json-converter
######################################################################
sys.path.insert(0, '/gpfs3/well/lindgren/users/swf744/git/image-to-coco-json-converter/')
from src.create_annotations_1channel_geoJSON import *
sys.path.insert(0, '/gpfs3/well/lindgren/users/swf744/git//PyTorchUnet/')




prs = argparse.ArgumentParser()
prs.add_argument('--inputFile', help='path of input masks', type=str)
args = vars(prs.parse_args())

assert args['inputFile'] != ""

# Label ids of the dataset
category_ids = {
    "background": 0,
    "adipocyte": 1
}

# Define which colors match which categories in the images
category_colors = {
    "0": 0, # Outlier
    "255": 1, # Window
}

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = []

pixel_god=255

# Get "images" and "annotations" info 
def images_annotations_info(cv2_img):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    # The mask image is *.png but the original image is *.jpg.
    # We make a reference to the original file in the COCO JSON file
    original_file_name = "" 

    w, h = np.shape(cv2_img)

    cv2_img[cv2_img==np.min(cv2_img)] = 0
    cv2_img[cv2_img==np.max(cv2_img)] = 255

    labels, no_objects = ndimage.label(cv2_img)
    props=regionprops(labels)
    regprop_size=np.array(list(props[i].area * 0.2500**2 for i in range (0, no_objects)))

    # "images" info 
    image = create_image_annotation(original_file_name, w, h, image_id)
    images.append(image)

    polygonList = []

    mask_image_open = Image.fromarray(cv2_img)

    sub_masks = create_sub_masks(mask_image_open, w, h, pixel_god)
    for color, sub_mask in sub_masks.items():
        category_id = category_colors[color]
        
        # "annotations" info
        polygons, segmentations = create_sub_mask_annotation(sub_mask)
        polygonList.append(polygons)
        
        # Check if we have classes that are a multipolygon
        if category_id in multipolygon_ids:
            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
            
            annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)
            
            annotations.append(annotation)
            annotation_id += 1
        else:
            for i in range(len(polygons)):

                print(np.array(polygons[i].exterior.coords).ravel().tolist())
                

                # takes both min and max x or y coordinate
                maxPoly=np.max(np.array(polygons[i].exterior.coords).ravel().tolist())
                minPoly=np.min(np.array(polygons[i].exterior.coords).ravel().tolist())
                # if min and max x or y coordinates of polygon are inside the box then whole polygon should be inside the box
                # this is to check if polygon is inside of the box (mid tile)
                inBox = (maxPoly > 1023 and maxPoly < 2045) and (minPoly > 1023 and minPoly < 2045)

                # draws up lines around the edges of the mid tile
                line0 = shapely.geometry.LineString([[1023, 1023], [2045, 1023]]) 
                line1 = shapely.geometry.LineString([[2045, 1023], [2045, 2045]]) 
                line2 = shapely.geometry.LineString([[2045, 2045], [1023, 2045]]) 
                line3 = shapely.geometry.LineString([[1023, 2045], [1023, 1023]]) 

                # check if polygons intersects any of these four edge lines, meaning is on the edge of the tile
                onLine = line0.intersects(polygons[i]) or line1.intersects(polygons[i]) or line2.intersects(polygons[i]) or line3.intersects(polygons[i])

                print('Edge lines intersect polgyon:'+str(onLine))
                print('Mid box has polygon inside:'+str(inBox))

                # Cleaner to recalculate this variable
                segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                
                annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                annotations.append(annotation)
                annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id, regprop_size, no_objects, polygonList

if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
            
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    x=8192
    y=(8192-1024)

    merged0 = np.concatenate((cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x-1024))+"_Y"+str((y-1024))+"_mask.png",0),
                              cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x))+"_Y"+str((y-1024))+"_mask.png",0),
                              cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x+1024))+"_Y"+str((y-1024))+"_mask.png",0)))

    merged1 = np.concatenate((cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x-1024))+"_Y"+str(y)+"_mask.png",0),
                              cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x))+"_Y"+str(y)+"_mask.png",0),
                              cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x+1024))+"_Y"+str(y)+"_mask.png",0)))

    merged2 = np.concatenate((cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x-1024))+"_Y"+str((y+1024))+"_mask.png",0),
                              cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x))+"_Y"+str((y+1024))+"_mask.png",0),
                              cv2.imread("/gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/tilesImageCollection_0000006277_2018-07-25zoomedProb08/tilesImageCollection_0000006277_2018-07-25ImageCollection_0000006277_2018-07-25_10_16_27_X"+str((x+1024))+"_Y"+str((y+1024))+"_mask.png",0)))

    totalMerged = np.concatenate((merged0, merged1, merged2),axis=1)

    totalMerged[totalMerged==np.min(totalMerged)] = 0
    totalMerged[totalMerged==np.max(totalMerged)] = 255
    
    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt, regprop_size, no_objects, polygonList = images_annotations_info(totalMerged)

    # for generating the geoJSON file
    # from https://gis.stackexchange.com/questions/130963/write-geojson-into-a-geojson-file-with-python
    # and https://stackoverflow.com/questions/55203858/how-to-format-geojson-file-from-python-dump-using-python-geojson
    features = []
    number = 0 
    for polygonEle in polygonList[0]:
        features.append(Feature(geometry=Polygon([(x,y) for x,y in polygonEle.exterior.coords]), properties={"id": str(number)}))
        number += 1

    feature_collection = FeatureCollection(features)

    with open(args['inputFile'].replace("_mask.png","_mask.geojson").split("/")[-1], "w") as f:
        dump(feature_collection, f)

    
    with open(args['inputFile'].replace("_mask.png","_mask.json").split("/")[-1],"w") as outfile:
        json.dump(coco_format, outfile)

    print("with regprops we get (mean size, no_objecs):")
    print(np.mean(regprop_size))
    print(no_objects)

    areas=[i['area'] * 0.2500**2 for i in coco_format["annotations"]]
    print("with polygons we get (mean size, no_objecs):")
    print(np.mean(areas))
    print(len(areas))


    print("with regprops we get (mean size, no_objecs) - after filtering:")
    regprop_size2=[i for i in regprop_size if i >= 200 and i <= 16000]
    print(np.mean(regprop_size2))
    print(len(regprop_size2))


    print("with polygons we get (mean size, no_objecs) - after filtering:")
    areas2=[i for i in areas if i >= 200 and i <= 16000]
    print(np.mean(areas2))
    print(len(areas2))


    maxRegprops=np.max(regprop_size)
    maxAreas=np.max(areas)

    bins = np.linspace(0, np.max((maxAreas,maxRegprops)), 10)

    pyplot.hist(regprop_size, bins, alpha=0.5, label='regionprops')
    pyplot.hist(areas, bins, alpha=0.5, label='polygons')
    pyplot.legend(loc='upper right')
    pyplot.savefig(args['inputFile'].replace("_mask.png","_hist.png").split("/")[-1])


    maxRegprops2=np.max(regprop_size2)
    maxAreas2=np.max(areas2)

    bins2 = np.linspace(0, np.max((maxAreas2,maxRegprops2)), 10)

    pyplot.hist(regprop_size2, bins2, alpha=0.5, label='regionprops')
    pyplot.hist(areas2, bins2, alpha=0.5, label='polygons')
    pyplot.legend(loc='upper right')
    pyplot.savefig(args['inputFile'].replace("_mask.png","_hist_filtered.png").split("/")[-1])



print(np.shape(totalMerged))
print(totalMerged)

## then we just need to be able to generate polygons of the masks
## store a centroid point for each polygon (absolute/WSI coordinates)
## then we have to see which ones intersect our middle tile
## remove those not in the middle that do not intersect, and that have been processed before
## and then get area estimates!

labels, no_objects = ndimage.label(totalMerged)
props=regionprops(labels)
size={i:props[i].area for i in range (0, no_objects)}

print(size)
print(no_objects)
