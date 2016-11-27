#!/bin/bash
# Perform Monte Carlo cross-validation (repeated random sub-sampling)
# on the face recognition system with a face database. The database
# should have the structure that is defined in ./scripts/create-sets.sh.

# parse arguments
EXT="pgm"

RUN_MATLAB=1
RUN_C=1

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
    -t|--num-test)
        NUM_TEST="$2"
        shift
        ;;
    -i|--num-iter)
        NUM_ITER="$2"
        shift
        ;;
    --matlab-only)
        RUN_C=0
        ;;
    --c-only)
        RUN_MATLAB=0
        ;;
    --pca)
        ARGS="$ARGS $1"
        PCA=1
        ;;
    --lda)
        ARGS="$ARGS $1"
        LDA=1
        ;;
    --ica)
        ARGS="$ARGS $1"
        ICA=1
        ;;
    -e|--loglevel)
        ARGS="$ARGS --loglevel $2"
        shift
        ;;
    -s|--timing)
        ARGS="$ARGS --timing"
        ;;
    --pca_n1)
        ARGS="$ARGS --pca_n1 $2"
        shift
        ;;
    --lda_n1)
        ARGS="$ARGS --lda_n1 $2"
        shift
	;;
    --lda_n2)
        ARGS="$ARGS --lda_n2 $2"
        shift
        ;;
    *)
        # unknown option
        ;;
    esac

    shift
done

if [[ -z $DB_PATH || -z $NUM_TEST || -z $NUM_ITER || ($PCA = 0 && $LDA = 0 && $ICA = 0) ]]; then
    >&2 echo "usage: ./scripts/cross-validate.sh [options]"
    >&2 echo
    >&2 echo "options:"
    >&2 echo "  -p, --path        path to image database"
    >&2 echo "  -t, --num-test N  number of samples to remove from training set"
    >&2 echo "  -i, --num-iter N  number of random iterations"
    >&2 echo "  --matlab-only     run MATLAB code only"
    >&2 echo "  --c-only          run C code only"
    >&2 echo "  --pca             run PCA"
    >&2 echo "  --lda             run LDA"
    >&2 echo "  --ica             run ICA"
    >&2 echo
    >&2 echo "options for C code:"
    >&2 echo "  --loglevel LEVEL  set the log level"
    >&2 echo "  --timing          print timing information"
    >&2 echo "  --pca_n1 N        (PCA) number of columns in W_pca to use"
    >&2 echo "  --lda_n1 N        (LDA) number of columns in W_pca to use"
    >&2 echo "  --lda_n2 N        (LDA) number of columns in W_fld to use"
    exit 1
fi

# build executables
if [ $RUN_C = 1 ]; then
    echo "Building..."
    echo

    make

    echo
fi

# determine the number of observations in each class
NUM_TRAIN=$(ls $DB_PATH/$(ls $DB_PATH | head -n 1) | wc -l)

# begin iterations
echo "Performing Monte Carlo cross-validation with p=$NUM_TEST and n=$NUM_ITER"
echo

for (( i = 1; i <= $NUM_ITER; i++ )); do
    # select p random observations
    SAMPLES=$(shuf -i 1-$NUM_TRAIN -n $NUM_TEST)
    SAMPLES=$(echo $SAMPLES | tr ' ' ',')

    echo "TEST $i: removing observations $SAMPLES from each class"
    echo

    # create the training set and test set
    ./scripts/create-sets.sh -p $DB_PATH -e $EXT -s $SAMPLES

    # run the algorithms
    if [ $RUN_MATLAB = 1 ]; then
        if [[ $RUN_MATLAB = 1 && $RUN_C = 1 ]]; then
            echo "MATLAB:"
        fi

        NUM_LINES=$((1 + PCA + LDA + ICA))

        matlab -nojvm -nodisplay -nosplash -r "cd MATLAB; face_rec('../train_images', '../test_images', $PCA, $LDA, $ICA, false); quit" | tail -n $NUM_LINES
    fi

    if [ $RUN_C = 1 ]; then
        if [[ $RUN_MATLAB = 1 && $RUN_C = 1 ]]; then
            echo "C:"
        fi

        ./face-rec --train train_images --test test_images $ARGS
    fi

    echo
done
