#!/bin/bash
# Create a training and test set from a face database
# by removing a range of observations from each class.
#
# The face database should have the following structure:
# database/
#   class1/
#     image1.[ext]
#     image2.[ext]
#     image3.[ext]
#     ...
#   class2/
#   class3/
#   ...
#
# EXAMPLES
#
# Remove 2.pgm from each class:
# ./scripts/create-sets.sh -p [db-path] -e pgm -s 2
#
# Remove 4.pgm - 7.pgm from each class:
# ./scripts/create-sets.sh -p [db-path] -e pgm -s 4,5,6,7
#
# TODO: maybe implement ranges with a dash (i.e. '4-7')

# parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    -p|--path)
        DB_PATH="$2"
        shift
        ;;
    -e|--ext)
        EXT="$2"
        shift
        ;;
    -s|--samples)
        SAMPLES="$2"
        shift
        ;;
    *)
        # unknown option
        ;;
    esac

    shift
done

if [[ -z $DB_PATH || -z $EXT || -z $SAMPLES ]]; then
    >&2 echo "usage: ./scripts/create-sets.sh [options]"
    >&2 echo
    >&2 echo "options:"
    >&2 echo "  -p, --path             path to image database"
    >&2 echo "  -e, --ext              image file extension"
    >&2 echo "  -s, --samples SAMPLES  comma-separated list of samples to remove from training set"
    exit 1
fi

# initialize training set and test set
rm -rf train_images test_images
cp -r $DB_PATH train_images
mkdir test_images

# transform samples argument into space-separated list
SAMPLES=$(echo $SAMPLES | tr ',' ' ')

# move specified observations from each class to the test set
for f in train_images/*; do
    for i in $SAMPLES; do
        mv $f/$i.$EXT test_images/$(basename $f)_$i.$EXT
    done
done
