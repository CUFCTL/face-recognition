#!/bin/bash
# Create a training and test set from a face database
# by removing a range of observations from each class.
#
# The face database should have the following structure:
# database/
#   class1/
#     1.[ext]
#     2.[ext]
#     3.[ext]
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
DB_PATH=$(basename $DB_PATH)
TEMP_PATH="$DB_PATH"_temp
TRAIN_PATH=train_images
TEST_PATH=test_images

rm -rf $TEMP_PATH $TRAIN_PATH $TEST_PATH
cp -r $DB_PATH $TEMP_PATH
mkdir $TRAIN_PATH $TEST_PATH

# transform samples argument into space-separated list
SAMPLES=$(echo $SAMPLES | tr ',' ' ')

# partition the data set into the training/test sets
for f in $TEMP_PATH/*; do
    # skip regular files (e.g. the README)
    if [ -f $f ]; then
        continue
    fi

    class=$(basename $f)

    # move test images
    for i in $SAMPLES; do
        mv $f/$i.$EXT $TEST_PATH/"$class"_$i.$EXT
    done

    # move training images
    for img in $f/*; do
        mv $img $TRAIN_PATH/"$class"_$(basename $img)
    done
done

rm -rf $TEMP_PATH
