#!/bin/sh
# Run an experiment on the dataset partition.

# define default settings
GPU=0
NUM_ITER=3

TEST_START=10
TEST_END=100
TEST_INC=10

# parse command-line arguments
while [[ $# -gt 0 ]]; do
	key="$1"

	case $key in
	-g|--gpu)
		GPU=1
		shift
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
		# unknown option
		;;
	esac

	shift
done

if [[ -z $DATASET || -z $ALGO ]]; then
	>&2 echo "usage: ./scripts/pbs/hyperparameter.sh [options]"
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
for (( N = $TEST_START; N <= $TEST_END; N += $TEST_INC )); do
	TRAIN=`expr 100 - $N`

	echo "Testing with $TRAIN/$N partition"

	python ./scripts/cross-validate.py -d $DATASET -t $TRAIN -r $N -i $NUM_ITER --$ALGO || exit -1
	echo
done
