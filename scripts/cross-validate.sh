#!/bin/bash
# Perform a k-fold cross-validation on the face recognition
# system with a face database. The database should have the
# structure that is defined in ./create-sets.sh.
#
# EXAMPLES
#
# Perform k-fold on a single observation (2):
# ./scripts/cross-validate.sh -p orl_faces -e pgm -r 2 2 [--lda --ica]
#
# Perform k-fold on a range of observations (4 - 7):
# ./scripts/cross-validate.sh -p orl_faces -e pgm -r 4 7 [--lda --ica]

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
    -r|--range)
        START="$2"
        shift
        END="$2"
        shift
        ;;
    --lda|--ica|--all)
        ARGS="$ARGS $1"
        ;;
    *)
        # unknown option
        ;;
    esac

    shift
done

if [[ -z $DB_PATH || -z $EXT || -z $START || -z $END ]]; then
    >&2 echo "usage: ./scripts/cross-validate.sh [options]"
    >&2 echo
    >&2 echo "options:"
    >&2 echo "  -p, --path             path to image database"
    >&2 echo "  -e, --ext              image file extension"
    >&2 echo "  -r, --range BEGIN END  range of samples to remove from training set"
    >&2 echo "  --lda                  run LDA"
    >&2 echo "  --ica                  run ICA"
    >&2 echo "  --all                  run all algorithms (PCA, LDA, ICA)"
    exit 1
fi

# build executables
make

echo "Performing k-fold cross-validation on the range [$START, $END]"
echo

for (( i = $START; i <= $END; i++ )); do
    echo "BEGIN: remove observation $i from each class"
    echo

    # create the training set and test set
    ./scripts/create-sets.sh -p $DB_PATH -e $EXT -r $i $i

    # run the algorithms
    ./face-rec --train train_images --rec test_images $ARGS
done
