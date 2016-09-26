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
# ./scripts/create-sets.sh [db-path] pgm 2 2
#
# Remove 4.pgm - 7.pgm from each class:
# ./scripts/create-sets.sh [db-path] pgm 4 7

# parse arguments
if [ "$#" -lt 4 ]; then
    >&2 echo "usage: ./scripts/create-sets.sh [db-path] [ext] [begin-index] [end-index]"
    exit 1
fi

DB_PATH=$1
EXT=$2
START=$3
END=$4

# initialize training set and test set
rm -rf train_images test_images
cp -r $DB_PATH train_images
mkdir test_images

# move range of observations from each class to the test set
for f in train_images/*; do
    for (( i = $START; i <= $END; i++ )); do
        mv $f/$i.$EXT test_images/$(basename $f)_$i.$EXT
    done
done
