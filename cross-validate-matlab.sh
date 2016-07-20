#!/bin/bash
# Cross validation for PCA, LDA, ICA, based on the ORL face
# database, which should be located at ./orl_faces.
#
# EXAMPLES
#
# Perform k-fold on a single observation (2.pgm):
# ./cross-validate.sh 2 2
#
# Perform k-fold on a range of observations (4.pgm - 7.pgm):
# ./cross-validate.sh 4 7
#
# Perform k-fold for all observations:
# ./cross-validate.sh 1 10

# parse arguments
if [ "$#" -lt 2 ]; then
    >&2 echo "usage: ./cross-validate-matlab.sh [begin-index] [end-index] [--pca --lda --ica]"
    exit 1
fi

DB_PATH=orl_faces
START=$1
END=$2
PCA=0
LDA=0
ICA=0

for i in "$@"; do
    if [ $i = "--pca" ]; then
        PCA=1
    elif [ $i = "--lda" ]; then
        LDA=1
    elif [ $i = "--ica" ]; then
        ICA=1
    fi
done

echo "Performing k-fold cross-validation on the range [$START, $END]"
echo

for (( i = $START; i <= $END; i++ )); do
    echo "BEGIN: remove $i.pgm from each class"
    echo

    # create the training set and test set
    rm -rf train_images_ppm test_images_ppm
    mkdir train_images_ppm test_images_ppm

    ./create-sets.sh $DB_PATH 1 10
    ./convert-images.sh test_images train_images_ppm pgm ppm > /dev/null

    mv train_images_ppm/*$i.ppm test_images_ppm

    # run the algorithms
    if [ $PCA = 1 ]; then
        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB/PCA; example; quit"
    fi

    if [ $LDA = 1 ]; then
        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB/LDA; example; quit"
    fi

    if [ $ICA = 1 ]; then
        echo "MATLAB ICA is not yet supported"
    fi
done
