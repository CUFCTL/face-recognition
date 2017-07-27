#!/bin/bash
# Run an experiment on the dataset partition.

# define default settings
GPU=0
NUM_ITER=3

TEST_START=10
TEST_END=90
TEST_INC=10

# parse command-line arguments
while [[ $# -gt 0 ]]; do
	key="$1"

	case $key in
	-g|--gpu)
		GPU=1
		;;
	-d|--dataset)
		DATASET="$2"
		shift
		;;
	-i|--num_iter)
		NUM_ITER="$2"
		shift
		;;
	-a|--algo)
		ALGO="$2"
		shift
		;;
	*)
		>&2 echo "error: unrecognized option '$1'"
		exit 1
		;;
	esac

	shift
done

if [[ -z $DATASET || -z $ALGO ]]; then
	>&2 echo "usage: ./partition.sh [options]"
	>&2 echo
	>&2 echo "options:"
	>&2 echo "  -g, --gpu       whether to run on GPU"
	>&2 echo "  -d, --dataset   dataset (feret, mnist, orl)"
	>&2 echo "  -i, --num_iter  number of iterations"
	>&2 echo "  -a, --algo      algorithm (pca, lda, ica)"
	exit 1
fi

# build executable
make GPU=$GPU > /dev/null

# run experiment
for (( TEST = $TEST_START; TEST <= $TEST_END; TEST += $TEST_INC )); do
	TRAIN=$((100 - TEST))
	RESULTS=$(python ./scripts/cross-validate.py -d $DATASET -t $TRAIN -r $TEST -i $NUM_ITER -- --$ALGO)

	echo $TRAIN/$TEST $RESULTS
done
