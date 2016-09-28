#!/bin/bash
# Perform Monte Carlo cross-validation (repeated random sub-sampling)
# on the MATLAB code with a face database. The database should have the
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
    -t|--num-test)
        NUM_TEST="$2"
        shift
        ;;
    -i|--num-iter)
        NUM_ITER="$2"
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

if [[ -z $DB_PATH || -z $EXT || -z $NUM_TEST || -z $NUM_ITER ]]; then
    >&2 echo "usage: ./scripts/cross-validate-matlab.sh [options]"
    >&2 echo
    >&2 echo "options:"
    >&2 echo "  -p, --path      path to image database"
    >&2 echo "  -e, --ext       image file extension"
    >&2 echo "  -t, --num-test  number of samples to remove from training set"
    >&2 echo "  -i, --num-iter  number of random iterations"
    >&2 echo "  --pca           run PCA"
    >&2 echo "  --lda           run LDA"
    >&2 echo "  --ica           run ICA"
    exit 1
fi

# determine the number of observations in each class
NUM_TRAIN=10

# begin iterations
echo "Performing Monte Carlo cross-validation with p=$NUM_TEST and n=$NUM_ITER"
echo

for (( i = 1; i <= $NUM_ITER; i++ )); do
    # select p random observations
    SAMPLES=$(shuf -i 1-$NUM_TRAIN -n $NUM_TEST)
    SAMPLES=$(echo $SAMPLES | tr ' ' ',')

    echo "BEGIN: removing observations $SAMPLES from each class"
    echo

    # create the training set and test set
    rm -rf train_images test_images
    cp -r $DB_PATH train_images
    mkdir test_images

    SAMPLES=$(echo $SAMPLES | tr ',' ' ')

    for j in $SAMPLES; do
        mv train_images/*$j.$EXT test_images
    done

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
