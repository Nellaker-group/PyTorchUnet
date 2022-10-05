import geojson 
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from shapely.ops import unary_union

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

## old version of this function
def same_area(poly0, poly1, thres=0.0001):
    return abs(poly0.area-poly1.area) < thres

## updated so this one also checks that the max position of X and Y is the same
def same_area_coords(poly0, poly1, thres=0.0001):
    same_coords = abs(poly0.bounds[0] - poly1.bounds[0]) < thres and abs(poly0.bounds[1] - poly1.bounds[1]) < thres
    same_area = abs(poly0.area-poly1.area) < thres
    return same_coords and same_area


def uniquify(masterList):
    ## we remove dupliace polygons, by removing polygons with the same area (within threshold)
    uniqpolies = []
    for poly in masterList:
        if not any(same_area_coords(p,poly) for p in uniqpolies):
            uniqpolies.append(poly) 
    return(uniqpolies)

def polygonMaskV2(geojsonOrg, geojsonShift):
     with open(geojsonOrg) as f0:
        gj0 = geojson.load(f0)
    f0.close()
     with open(geojsonShift) as f1:
        gj1 = geojson.load(f1)
    f1.close()
    masterList = geojson2polygon(geojson0)
    newList = geojson2polygon(geojson1)
    for new_poly in newList:
        masterList = merge_polys(new_poly, masterList)
    masterList = uniquify(masterList)
    return(masterList)
