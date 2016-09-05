#!/bin/bash
# Perform a k-fold cross-validation on the MATLAB code
# with a face database. The database should have the
# following structure:
#
# database/
#  class1_image1.ppm
#  class1_image2.ppm
#  class1_imageN.ppm
#  ...
#  class2_image1.ppm
#  class2_image2.ppm
#  class2_imageN.ppm
#  ...
#
# EXAMPLES
#
# Perform k-fold on a single observation (2):
# ./cross-validate-matlab.sh orl_faces_ppm 2 2 [--pca --lda --ica]
#
# Perform k-fold on a range of observations (4 - 7):
# ./cross-validate-matlab.sh orl_faces_ppm 4 7 [--pca --lda --ica]

# parse arguments
if [ "$#" -lt 3 ]; then
    >&2 echo "usage: ./cross-validate-matlab.sh [db-path] [begin-index] [end-index] [--pca --lda --ica]"
    exit 1
fi

DB_PATH=$1
EXT=ppm
START=$2
END=$3
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
    echo "BEGIN: remove observation $i from each class"
    echo

    # create the training set and test set
    rm -rf train_images test_images
    cp -r $DB_PATH train_images
    mkdir test_images
    mv train_images/*$i.ppm test_images

    # run the algorithms
    if [ $PCA = 1 ]; then
        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB/PCA; example; quit"
    fi

    if [ $LDA = 1 ]; then
        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB/LDA; example; quit"
    fi

    if [ $ICA = 1 ]; then
        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB/ICA_new; run_ica; quit"
    fi
done
