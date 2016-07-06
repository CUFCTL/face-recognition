#!/bin/bash
# Convert a directory of images from one image format to another.
#
# EXAMPLES
#
# Convert JPG images in './images' to PGM images in './images2':
# ./convert-images.sh ./images ./images2 jpg pgm

# parse arguments
if [ "$#" = 4 ]; then
    SRC_DIR=$1
    DST_DIR=$2
    SRC_FMT=$3
    DST_FMT=$4
else
    >&2 echo "usage: ./convert-images.sh [src-path] [dst-path] [src-format] [dst-format]"
    exit 1
fi

# convert images
mkdir -p $DST_DIR

for f in $SRC_DIR/*.$SRC_FMT; do
    echo "$f -> $DST_DIR/$(basename $f .$SRC_FMT).$DST_FMT"
    convert $f "$DST_DIR/$(basename $f .$SRC_FMT).$DST_FMT"
done
