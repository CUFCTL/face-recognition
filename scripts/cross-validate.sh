#!/bin/bash
# Perform Monte Carlo cross-validation (repeated random sub-sampling)
# on the face recognition system with a face database. The database
# should have the structure that is defined in ./scripts/create-sets.sh.

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
    -t|--num-test)
        NUM_TEST="$2"
        shift
        ;;
    -i|--num-iter)
        NUM_ITER="$2"
        shift
        ;;
    --pca|--lda|--ica|--all)
        ARGS="$ARGS $1"
        ;;
    *)
        # unknown option
        ;;
    esac

    shift
done

if [[ -z $DB_PATH || -z $EXT || -z $NUM_TEST || -z $NUM_ITER ]]; then
    >&2 echo "usage: ./scripts/cross-validate.sh [options]"
    >&2 echo
    >&2 echo "options:"
    >&2 echo "  -p, --path      path to image database"
    >&2 echo "  -e, --ext       image file extension"
    >&2 echo "  -t, --num-test  number of samples to remove from training set"
    >&2 echo "  -i, --num-iter  number of random iterations"
    >&2 echo "  --pca           run PCA"
    >&2 echo "  --lda           run LDA"
    >&2 echo "  --ica           run ICA"
    >&2 echo "  --all           run all algorithms (PCA, LDA, ICA)"
    exit 1
fi

# build executables
make

# determine the number of observations in each class
NUM_TRAIN=$(ls $DB_PATH/$(ls $DB_PATH | head -n 1) | wc -l)

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
    ./scripts/create-sets.sh -p $DB_PATH -e $EXT -s $SAMPLES

    # run the algorithms
    ./face-rec --train train_images --rec test_images $ARGS
done
