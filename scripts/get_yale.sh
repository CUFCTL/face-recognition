#!/bin/bash
# Download and extract the Yale face database.

mkdir -p datasets
cd datasets

# wget http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip
unzip yalefaces.zip > /dev/null

rm -rf __MACOSX

# clean up image names
mv yalefaces/subject01.gif yalefaces/subject01.centerlight
rm yalefaces/subject01.glasses.gif

for f in yalefaces/subject*; do
	mv $f $f.gif
done

# convert images to PGM
cd ..
./scripts/convert-images.sh datasets/yalefaces datasets/yalefaces_pgm gif pgm > /dev/null

rm -rf datasets/yalefaces
mv datasets/yalefaces_pgm datasets/yalefaces
