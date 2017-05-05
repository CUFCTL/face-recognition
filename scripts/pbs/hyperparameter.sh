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
	--start)
		TEST_START="$2"
		shift
		;;
	--end)
		TEST_END="$2"
		shift
		;;
	--inc)
		TEST_INC="$2"
		shift
		;;
	*)
		>&2 echo "error: unrecognized option '$1'"
		exit 1
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
	>&2 echo "  --start         hyperparameter start"
	>&2 echo "  --end           hyperparameter stop"
	>&2 echo "  --inc           hyperparameter increment"
	exit 1
fi

# generate range of values
if [ $PARAM = "ica_nonl" ]; then
	VALUES="pow3 tanh gauss"
elif [ $PARAM = "knn_dist" ]; then
	VALUES="L1 L2 COS"
else
	for (( i = $TEST_START; i <= $TEST_END; i += $TEST_INC )); do
		VALUES="$VALUES $i"
	done
fi

# build executable
make GPU=$GPU > /dev/null

# run experiment
for N in $VALUES; do
	echo "Testing with $PARAM = $N"

	python ./scripts/cross-validate.py -d $DATASET -t $TRAIN -r $TEST -i $NUM_ITER --$ALGO -- --$PARAM $N
	echo
done
