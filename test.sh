#!/bin/bash
# Leave-one-out cross validation for PCA, LDA, ICA
# TODO: k-fold cross validation

if [ "$#" -ne 1 ]; then
    echo "usage: ./test.sh [test-index]"
    exit
fi

# build executables
make

# create training set and test set
rm -rf train_images test_images
cp -r orl_faces train_images
mkdir test_images

for f in train_images/*
    do mv $f/$1.pgm test_images/$(basename $f)_$1.pgm
done

# run the algorithms
./train train_images
./recognize test_images
