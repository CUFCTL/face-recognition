#!/bin/bash
# Run an experiment on the feature-classifier algorithm.

# define default settings
GPU=0
TRAIN=70
TEST=30
NUM_ITER=3

FEATURES="pca lda ica"
CLASSIFIERS="knn bayes"

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
	-t|--train)
		TRAIN="$2"
		shift
		;;
	-r|--test)
		TEST="$2"
		shift
		;;
	-i|--num_iter)
		NUM_ITER="$2"
		shift
		;;
	*)
		>&2 echo "error: unrecognized option '$1'"
		exit 1
		;;
	esac

	shift
done

if [[ -z $DATASET ]]; then
	>&2 echo "usage: ./algorithm.sh [options]"
	>&2 echo
	>&2 echo "options:"
	>&2 echo "  -g, --gpu       whether to run on GPU"
	>&2 echo "  -d, --dataset   dataset (feret, mnist, orl)"
	>&2 echo "  -t, --train     training partition (0-100)"
	>&2 echo "  -r, --test      testing partition (0-100)"
	>&2 echo "  -i, --num_iter  number of iterations"
	exit 1
fi

# build executable
make GPU=$GPU > /dev/null

# run experiment
for C in $CLASSIFIERS; do
	for F in $FEATURES; do
		RESULTS=$(python ./scripts/cross-validate.py -d $DATASET -t $TRAIN -r $TEST -i $NUM_ITER -- --$F --$C)

		echo $F $C $RESULTS
	done
done
