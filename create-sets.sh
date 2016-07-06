#!/bin/bash
# Create a training and test set from the ORL face database
# by removing a range of observations from each class.
#
# EXAMPLES
#
# Remove 2.pgm from each class:
# ./create-sets.sh 2
#
# Remove 4.pgm - 7.pgm from each class:
# ./create-sets.sh 4 7

# determine the range
if [ "$#" = 2 ]; then
    START=$1
    END=$2
elif [ "$#" = 1 ]; then
    START=$1
    END=$1
else
    >&2 echo "usage: ./create-sets.sh [begin-index] [end-index]"
    exit 1
fi

# initialize training set and test set
rm -rf train_images test_images
cp -r orl_faces train_images
mkdir test_images

# move range of observations from each class to the test set
for f in train_images/*; do
    for (( i = $START; i <= $END; i++ )); do
        mv $f/$i.pgm test_images/$(basename $f)_$i.pgm
    done
done
