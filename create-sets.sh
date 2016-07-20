#!/bin/bash
# Create a training and test set from a face database
# by removing a range of observations from each class.
#
# The face database should have the following structure:
# database/
#   class1/
#     image1.pgm
#     image2.pgm
#     image3.pgm
#     ...
#   class2/
#   class3/
#   ...
#
# EXAMPLES
#
# Remove 2.pgm from each class:
# ./create-sets.sh [db-path] 2 2
#
# Remove 4.pgm - 7.pgm from each class:
# ./create-sets.sh [db-path] 4 7

# parse arguments
if [ "$#" -lt 3 ]; then
    >&2 echo "usage: ./create-sets.sh [db-path] [begin-index] [end-index]"
    exit 1
fi

DB_PATH=$1
START=$2
END=$3

# initialize training set and test set
rm -rf train_images test_images
cp -r $DB_PATH train_images
mkdir test_images

# move range of observations from each class to the test set
for f in train_images/*; do
    for (( i = $START; i <= $END; i++ )); do
        mv $f/$i.pgm test_images/$(basename $f)_$i.pgm
    done
done
