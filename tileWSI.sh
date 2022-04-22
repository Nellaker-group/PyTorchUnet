#!/bin/bash

## apparently .csn files (Leica files) are created according to the dimensions of miscropcopse
## however often the actual image is much smaller than that meaning, there is a lot of empty space in the image
## to get to know the location of the actual image within this one has to run openslide-show-properties 
## and then look at openslide.bounds-x and openslide.bounds-y
## They are The X and Y coordinate of the rectangle bounding the non-empty region of the slide.
## More can also be seen here:
## https://github.com/openslide/openslide-python/issues/48

## WHEN THERE ARE MORE IMAGES IN .scn FILE, I AM NOT SURE WHICH ONE IS READ WITH OPENSLIDE IN PYTHON??

## module load OpenSlide/3.4.1-GCCcore-8.2.0-largefiles

FILE="$1"
DIR="$2"
IF512="$3"

echo $FILE

if [[ $FILE == *.scn ]]
 then
    INFO1=`openslide-show-properties "$FILE" | egrep "openslide.bounds-x:|openslide.bounds-y:|openslide.region\[0\].height:|openslide.region\[0\].width:" | head -n 1 | cut -f2 -d" " | sed -e "s/'//g"`
    INFO2=`openslide-show-properties "$FILE" | egrep "openslide.bounds-x:|openslide.bounds-y:|openslide.region\[0\].height:|openslide.region\[0\].width:" | head -n 2 | tail -n 1 | cut -f2 -d" " | sed -e "s/'//g"`
    INFO3=`openslide-show-properties "$FILE" | egrep "openslide.bounds-x:|openslide.bounds-y:|openslide.region\[0\].height:|openslide.region\[0\].width:" | head -n 3 | tail -n 1 | cut -f2 -d" " | sed -e "s/'//g"`
    INFO4=`openslide-show-properties "$FILE" | egrep "openslide.bounds-x:|openslide.bounds-y:|openslide.region\[0\].height:|openslide.region\[0\].width:" | head -n 4 | tail -n 1 | cut -f2 -d" " | sed -e "s/'//g"`
else
    INFO1=0
    INFO2=0
    INFO3=0
    INFO4=0
fi

echo $INFO1
echo $INFO2
echo $INFO3
echo $INFO4

python tileWSI.py "$FILE" $INFO1 $INFO2 $INFO3 $INFO4 "$DIR" "$IF512"
