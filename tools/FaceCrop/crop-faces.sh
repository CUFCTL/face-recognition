#!/bin/bash
# Cross validation for PCA, LDA, ICA, based on the ORL face
# database, which should be located at ./orl_faces.
#
# EXAMPLES
#

# determine the path of images to be cropped
if [ "$#" = 1 ]; then
    path=$1
else
    >&2 echo "usage: ./crop-faces.sh ./path/to/test/set"
    exit 1
fi

rm -rf cropped_test_set
mkdir cropped_test_set

make clean

make

i=0

pwd

# move range of observations from each class to the test set
for f in $path/*/; do
    echo $f
    mkdir cropped_test_set/crop_$i
    ./detect $f ./cropped_test_set/crop_$i
    ((i++))
done
