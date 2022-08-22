import glob
import argparse
import cv2
import os
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from matplotlib import pyplot
from geojson import Point, Feature, FeatureCollection, dump
import time
import math

from shapely.ops import unary_union
from create_annotations_1channel_geoJSONV2 import *

prs = argparse.ArgumentParser()
prs.add_argument('--inputDirOrg', help='input directory of folders with mask files from org tiling', type=str)
prs.add_argument('--inputDirShift', help='input directory of folders with mask files from shifted tiling', type=str)
args = vars(prs.parse_args())
assert args['inputDirOrg'] != "" and args['inputDirShift'] != ""

# Label ids of the dataset
category_ids = {
    "background": 0,
    "adipocyte": 1
}

# Define which colors match which categories in the images
category_colours = {
    "background": 30, # Outlier
    "adipocyte": 215 # Window
}

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = []

pixel_god=215

# Emil helper function for getting absolute path of files in directory
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

# Get "images" and "annotations" info
def images_annotations_info(maskpath,X,Y):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    # The mask image is *.png but the original image is *.jpg.
    # We make a reference to the original file in the COCO JSON file
    original_file_name = maskpath.replace("_mask","")
    # Open the image and (to be sure) we convert it to RGB
    mask_image_open = Image.open(maskpath).convert("L")
    w, h = mask_image_open.size
    cv2_img = cv2.imread(maskpath,0)
    cv2_img[cv2_img==np.min(cv2_img)] = 0
    cv2_img[cv2_img==np.max(cv2_img)] = 255
    labels, no_objects = ndimage.label(cv2_img)
    props=regionprops(labels)
    regprop_size=np.array(list(props[i].area for i in range (0, no_objects)))
    # "images" info
    image = create_image_annotation(original_file_name, w, h, image_id)
    images.append(image)
    polygonList = []
    sub_masks = create_sub_masks(mask_image_open, w, h, pixel_god)
    for colour, sub_mask in sub_masks.items():
        category_id = category_colours[colour]
        # "annotations" info
        polygons, segmentations = create_sub_mask_annotation(sub_mask,X,Y)
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
                # Cleaner to recalculate this variable
                segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                annotations.append(annotation)
                annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id, regprop_size, no_objects, polygonList



# Get "images" and "annotations" info
def images_annotations_infoV2(maskpath,X,Y):
    original_file_name = maskpath.replace("_mask","")
    # Open the image and (to be sure) we convert it to RGB
    mask_image_open = Image.open(maskpath).convert("L")
    image01=np.array(mask_image_open)
    ## EMIL DO NOT HAVE HARD-CODED VALUES HERE
    image01[ image01 == category_colours["adipocyte"]] = 1
    image01[ image01 == category_colours["background"]] = 0    
    polygons, segmentations = create_sub_mask_annotation_simple(image01,X,Y)    
    return(polygons, segmentations)  

def merge_polys(new_poly, all_polys):
    all_polys_list = []
    for existing_poly in all_polys:
        if new_poly.intersects(existing_poly):
            new_poly = unary_union([new_poly, existing_poly])
            all_polys_list = merge_polys(new_poly, all_polys_list)
            all_polys_list.append(new_poly)
        else:
            all_polys_list.append(existing_poly)
    return all_polys_list

def same_area(poly0, poly1, thres=0.0001):
    return abs(poly0.area-poly1.area) < thres

def uniquify(masterList):
    ## we remove dupliace polygons, by removing polygons with the same area (within threshold)
    uniqpolies = []
    for poly in masterList:
        if not any(same_area(p,poly) for p in uniqpolies):
            uniqpolies.append(poly) 
    return(uniqpolies)


def polygonMask(filedir,masterList,first):
    onlyfiles = listdir_fullpath(filedir)        
    ## go through mask files of original tiling and create polygons
    tmpList = []
    for maskFile in onlyfiles:        
        X = int(maskFile.split("/")[-1].split("_X")[1].split("_Y")[0])
        Y = int(maskFile.split("/")[-1].split("_Y")[1].split("_mask")[0])
        print(maskFile)
        print("X is:")
        print(X)
        print("Y is:")
        print(Y)
        ## Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt, regprop_size, no_objects, polygonList = images_annotations_info(maskFile,X,Y)
        if not first:
            for new_poly in polygonList[0]:
                masterList = merge_polys(new_poly, masterList)
        else:
            masterList.append(polygonList[0])
    if first:        
        masterList = [x for xs in masterList for x in xs]
    return(masterList)
    ## make a flat master list of polygons
    if len(masterList)==0:
        newMasterList = tmpList
    else:
        newMasterList = [masterList,tmpList]
    print(newMasterList[0:2])
    print(newMasterList[(len(newMasterList)-2):len(newMasterList)])
    masterListFlat = [x for xs in newMasterList for x in xs]
    return(masterListFlat)


def polygonMaskV2(filedir,masterList,first):
    print("now going through this dir:")
    print(filedir)
    onlyfiles = listdir_fullpath(filedir)        
    ## go through mask files of original tiling and create polygons
    tmpList = []
    for maskFile in onlyfiles:        
        X = int(maskFile.split("/")[-1].split("_X")[1].split("_Y")[0])
        Y = int(maskFile.split("/")[-1].split("_Y")[1].split("_mask")[0])
        ## Create images and annotations sections
        segmentation, polygonList = images_annotations_infoV2(maskFile,X,Y)
        if not first:
            for new_poly in polygonList:
                masterList = merge_polys(new_poly, masterList)
        else:
            masterList.append(polygonList)
    if first:        
        masterList = [x for xs in masterList for x in xs]
    return(masterList)

def polsbyPopperTest(polygon):
    return((4*math.pi*polygon.area ) / ( polygon.length**2 ))
    
def polsbyPopperHist(polygonList,inputdir):
    tmpList = []
    for poly in polygonList:
        tmpList.append(polsbyPopperTest(poly))    
    pyplot.hist(tmpList, bins=10, alpha=0.5, range=(0,1))
    if "/" in inputdir:
        pyplot.savefig(list(filter(None,inputdir.split("/")))[-1]+"_polsbyPepperHist.png")
    else:
        pyplot.savefig(inputdir+"polsbyPepperHist.png")
    
def writeToGeoJSON(masterList, filename):
    number=0
    features = []
    for polygonEle in masterList: 
        if polygonEle.type == 'Polygon':
            features.append(Feature(geometry=Polygon([(x,y) for x,y in polygonEle.exterior.coords]), properties={"id": str(number)}))
        if polygonEle.type == 'MultiPolygon':            
            mycoordslist = [list(x.exterior.coords) for x in polygonEle.geoms]
            ll=[x for xs in mycoordslist for x in xs]
            features.append(Feature(geometry=Polygon(ll), properties={"id": str(number)}))
        number+=1
    feature_collection = FeatureCollection(features)        
    ## I add another key to the FeatureCollection dictionary where it can look up max ID, so it knows which ID to work on when adding elements
    feature_collection["maxID"]=number
    with open(filename,"w") as outfile:
        dump(feature_collection, outfile) 
    outfile.close()

def addToGeoJSON(geoJSONfile,toAdd):
    with open(geoJSONfile) as f:
        gj = geojson.load(f)
    number = gj["maxID"]
    if toAdd.type == 'Polygon':
        gj['features'].append(Feature(geometry=Polygon([(x,y) for x,y in toAdd.exterior.coords]), properties={"id": str(number)}))
    if toAdd.type == 'MultiPolygon':
        mycoordslist = [list(x.exterior.coords) for x in toAdd.geoms]
        ll=[x for xs in mycoordslist for x in xs]
        gj['features'].append(Feature(geometry=Polygon(ll), properties={"id": str(number)}))
    newname= geoJSONfile.replace(".geojson","_ADDED.geojson")
    with open(newname,"w") as outfile:
        dump(gj, outfile)
    outfile.close()   
                
if __name__ == "__main__":
    mask_path = "dataset/{}_mask/".format("val")
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)            
    ## go through files of org tiling

    masterList = []  
    t0 = time.time()
    masterList = polygonMaskV2(args['inputDirOrg'],masterList,True)
    t1 = time.time()
    print("First analysis finished:" + str((t1-t0)))
    masterList = polygonMaskV2(args['inputDirShift'],masterList,False)
    t2 = time.time()
    print("Second analysis finished:" + str((t2-t0)))
    masterList = uniquify(masterList)
    t3 = time.time()
    print("Third analysis finished:" + str((t3-t0)))
    writeToGeoJSON(masterList,"mergedSegmentationV3.geojson")
    polsbyPopperHist(masterList,args['inputDirOrg'])
     
    ## masterList = []        
    ## masterList = polygonMask(args['inputDir']+"org",masterList,True)
    ## writeToGeoJSON(masterList,"originalSegmentation.geojson")
    ## for new_poly in masterList:
    ##     masterList = merge_polys(new_poly, masterList)    
    ## writeToGeoJSON(masterList,"originalSegmentationMerged.geojson")
    ## masterList = polygonMask(args['inputDir']+"X512_Y512",masterList,False)
    ## masterList = uniquify(masterList)
    ## masterList = polygonMask(args['inputDir']+"X512_Y0",masterList,False)
    ## masterList = uniquify(masterList)
    ## masterList = polygonMask(args['inputDir']+"X0_Y512",masterList,False)
    ## masterList = uniquify(masterList)
    ## writeToGeoJSON(masterList,"mergedSegmentation.geojson")
    
    

                                           
