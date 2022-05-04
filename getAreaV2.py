from skimage.measure import regionprops
import cv2
import numpy as np
from scipy import ndimage
import pandas as pd
import argparse
import os

prs = argparse.ArgumentParser()
prs.add_argument('--inputDir', help='path of input masks', type=str)
prs.add_argument('--dataset', help='which dataset GTEX, ENDOX, MOBB or fatDIVA input masks', type=str)
args = vars(prs.parse_args())

assert args['inputDir'][-1] != "/"
assert args['dataset'] in ['GTEX', 'ENDOX', 'MOBB', 'fatDIVA']

def predict_areas(input_img,dataset):
    # counts and labels objects or seperate segmentation blobs                                                                                                      
    labels, no_objects = ndimage.label(input_img)
    assert np.shape(input_img) == np.shape(labels)
    # gives an object with the size of each blob                                                                                                                    
    props=regionprops(labels)
    # gets the size of each blob                                                                                                                                    
    size={i:props[i].area for i in range (0, no_objects)}
    # cuts away to small blobs and too large blobs                                                                                                                  
    # converts pixels into micrometers or something I guess from the number fro QuPath                                                                              
    if dataset == "GTEX":
        raw_areas=np.array(list(props[i].area for i in range (0, no_objects))) * 0.4942**2
    elif dataset == "ENDOX":
        raw_areas=np.array(list(props[i].area for i in range (0, no_objects))) * 0.2500**2
    elif dataset == "MOBB":
        pass
    elif dataset == "fatDIVA":
        pass
    no_of_cells=(sum(i >= 200 and i <= 16000 for i in raw_areas))
    areas=[i for i in raw_areas if i >= 200 and i <= 16000]
    if(no_of_cells==0):
        return(np.nan,np.nan,np.nan,no_of_cells,np.nan,no_objects)
    else:
        return(areas,np.mean(areas),np.std(areas),no_of_cells,raw_areas,no_objects)

files = os.listdir(args['inputDir']) 

listie=[]

for filename in files:
    ex_img = cv2.imread(args['inputDir']+"/"+filename,0)    
    ex_img[ex_img==np.min(ex_img)] = 0
    ex_img[ex_img==np.max(ex_img)] = 255
    cell_areas,mu_area,sd_area,no_cells,raw_areas,no_objects = predict_areas(ex_img,args['dataset'])
    listie.append((filename,mu_area,sd_area,no_cells))

df = pd.DataFrame(listie, columns=['filename', 'mu_area', 'sd_area', 'no_cells'])

df.to_csv(args['inputDir'].split("/")[-1]+"areaSize.csv",na_rep="nan",index=False)


print("total mean is:")
print(np.mean(df['mu_area']))
print(np.nanmean(df['mu_area']))

print("mean sd is:")
print(np.mean(df['sd_area']))
print(np.nanmean(df['sd_area']))

print("mean number of cells is:")
print(np.mean(df['no_cells']))
print(np.nanmean(df['no_cells']))

print("mean number of cells is (with tiles with cells > 0):")
print(np.mean(df[ df['no_cells']>0]['no_cells']))
print(np.nanmean(df[ df['no_cells']>0]['no_cells']))
