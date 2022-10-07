import geojson 
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from shapely.ops import unary_union


# new updated version from Chris 07-10-2022
def merge_polys(new_poly, all_polys):
    all_polys_list = []
    intersect_flag = False
    emptyPoly = Polygon([[-10, -10], [-10, -10], [-10, -10]])
    if all_polys == []: all_polys = [emptyPoly]
    for existing_poly in all_polys:        
        if new_poly.intersects(existing_poly):
            intersect_flag = True
            new_poly = unary_union([new_poly, existing_poly])            
            all_polys_list = merge_polys(new_poly, all_polys_list)
        else:  
            all_polys_list.append(existing_poly)
    if not intersect_flag: 
        all_polys_list.append(new_poly)
    return all_polys_list


def polygonMaskV2(geojsonOrg, geojsonShift):
     with open(geojsonOrg) as f0:
        gj0 = geojson.load(f0)
    f0.close()
     with open(geojsonShift) as f1:
        gj1 = geojson.load(f1)
    f1.close()
    masterList = [gj0, gj1]
    masterList = [x for xs in masterList for x in xs]
    newList = geojson2polygon(masterList)
    targetList = []
    for new_poly in newList:
        targetList = merge_polys(new_poly, targetList)
    return(targetList)
