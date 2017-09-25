#!/bin/bash
# Convert a directory of images from one image format to another.
#
# EXAMPLES
#
# Convert JPG images in './images' to PGM images in './images2':
# ./scripts/convert-images.sh ./images ./images2 jpg pgm

# parse arguments
if [ "$#" = 4 ]; then
    SRC_DIR=$1
    DST_DIR=$2
    SRC_FMT=$3
    DST_FMT=$4
else
    >&2 echo "usage: $0 [src-path] [dst-path] [src-format] [dst-format]"
    exit 1
fi

# convert images
mkdir -p $DST_DIR

for infile in $SRC_DIR/*.$SRC_FMT; do
    outfile=$DST_DIR/$(basename "$infile" .$SRC_FMT).$DST_FMT

    echo "$infile -> $outfile"
    convert "$infile" "$outfile"
done
