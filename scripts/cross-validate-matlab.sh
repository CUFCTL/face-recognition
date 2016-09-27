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
# ./scripts/cross-validate-matlab.sh -p orl_faces_ppm -r 2 2 [--pca --lda --ica]
#
# Perform k-fold on a range of observations (4 - 7):
# ./scripts/cross-validate-matlab.sh -p orl_faces_ppm -r 4 7 [--pca --lda --ica]

# parse arguments
EXT="ppm"
PCA=0
LDA=0
ICA=0

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
    -r|--range)
        START="$2"
        shift
        END="$2"
        shift
        ;;
    --pca)
        PCA=1
        ;;
    --lda)
        LDA=1
        ;;
    --ica)
        ICA=1
        ;;
    *)
        # unknown option
        ;;
    esac

    shift
done

if [[ -z $DB_PATH || -z $EXT || -z $START || -z $END ]]; then
    >&2 echo "usage: ./scripts/cross-validate-matlab.sh [options]"
    >&2 echo
    >&2 echo "options:"
    >&2 echo "  -p, --path             path to image database"
    >&2 echo "  -e, --ext              image file extension"
    >&2 echo "  -r, --range BEGIN END  range of samples to remove from training set"
    >&2 echo "  --pca                  run PCA"
    >&2 echo "  --lda                  run LDA"
    >&2 echo "  --ica                  run ICA"
    exit 1
fi

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
        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB/ICA; run_ica; quit"
    fi
done
