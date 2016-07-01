#!/bin/bash
# Cross validation for PCA, LDA, ICA, based on the ORL face
# database, which should be located at ./orl_faces.
#
# EXAMPLES
#
# Perform k-fold on a single observation (2.pgm):
# ./test.sh 2
#
# Perform k-fold on a range of observations (4.pgm - 7.pgm):
# ./test.sh 4 7
#
# Perform k-fold for all observations:
# ./test.sh 1 10

# determine the range
if [ "$#" = 2 ]; then
    START=$1
    END=$2
elif [ "$#" = 1 ]; then
    START=$1
    END=$1
else
    >&2 echo "usage: ./test.sh [begin-index] [end-index]"
    exit 1
fi

# build executables
make

echo "Performing k-fold cross-validation on the range [$START, $END]"
echo

for (( i = $START; i <= $END; i++ )); do
    echo "BEGIN: remove $i.pgm from each class"
    echo

    # initialize training set and test set
    rm -rf train_images test_images
    cp -r orl_faces train_images
    mkdir test_images

    # move the i-th image of each class to the test set
    for f in train_images/*
        do mv $f/$i.pgm test_images/$(basename $f)_$i.pgm
    done

    # run the algorithms
    ./train train_images
    ./recognize test_images
done
