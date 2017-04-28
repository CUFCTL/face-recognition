#!/bin/sh
# Run an experiment on a hyperparameter.

# define default settings
GPU=0
TRAIN=70
TEST=30
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
	-a|--algo)
		ALGO="$2"
		shift
		;;
	-p|--param)
		PARAM="$2"
		shift
		;;
	*)
		# unknown option
		;;
	esac

	shift
done

if [[ -z $DATASET || -z $ALGO || -z $PARAM ]]; then
	>&2 echo "usage: ./scripts/pbs/hyperparameter.sh [options]"
	>&2 echo
	>&2 echo "options:"
	>&2 echo "  -g, --gpu       whether to run on GPU"
	>&2 echo "  -d, --dataset   dataset (feret, mnist, orl)"
	>&2 echo "  -t, --train     training partition (0-100)"
	>&2 echo "  -r, --test      testing partition (0-100)"
	>&2 echo "  -i, --num_iter  number of iterations"
	>&2 echo "  -a, --algo      algorithm (pca, lda, ica)"
	>&2 echo "  -p, --param     hyperparameter"
	exit 1
fi

# build executable
make clean && make GPU=$GPU

# run experiment
for (( N = $TEST_START; N <= $TEST_END; N += $TEST_INC )); do
	echo "Testing with $PARAM = $N"
	python ./scripts/cross-validate.py --run-c -d $DATASET -t $TRAIN -r $TEST -i $NUM_ITER --$ALGO -- --$PARAM $N
done
