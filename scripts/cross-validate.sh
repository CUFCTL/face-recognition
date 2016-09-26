#!/bin/bash
# Perform a k-fold cross-validation on the face recognition
# system with a face database. The database should have the
# structure that is defined in ./create-sets.sh.
#
# EXAMPLES
#
# Perform k-fold on a single observation (2):
# ./scripts/cross-validate.sh orl_faces pgm 2 2
#
# Perform k-fold on a range of observations (4 - 7):
# ./scripts/cross-validate.sh orl_faces pgm 4 7

# parse arguments
if [ "$#" -lt 4 ]; then
    >&2 echo "usage: ./scripts/cross-validate.sh [db-path] [ext] [begin-index] [end-index] [--lda --ica --all]"
    exit 1
fi

DB_PATH=$1
EXT=$2
START=$3
END=$4

shift 4
ARGS="$@"

# build executables
make

echo "Performing k-fold cross-validation on the range [$START, $END]"
echo

for (( i = $START; i <= $END; i++ )); do
    echo "BEGIN: remove observation $i from each class"
    echo

    # create the training set and test set
    ./scripts/create-sets.sh $DB_PATH $EXT $i $i

    # run the algorithms
    ./face-rec --train train_images --rec test_images $ARGS
done
