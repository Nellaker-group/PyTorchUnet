#!/bin/bash

FILE="$1"
DIR="$2"
IF512="$3"
ZOOMFILE="$4"
SOURCEDATASET="$5"
## supposed to be like this 512,512 - how much you shift the frame for tiling
SHIFT="$6"

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

python tileWSI.py "$FILE" $INFO1 $INFO2 $INFO3 $INFO4 "$DIR" "$IF512" "$ZOOMFILE" "$SOURCEDATASET" $SHIFT
