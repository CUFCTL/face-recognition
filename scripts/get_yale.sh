#!/bin/bash
# Download and extract the Yale face database.

mkdir -p datasets
cd datasets

# download archive if necessary
if [ ! -f yalefaces.zip ]; then
	wget http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip
fi

# extract archive
rm -rf yalefaces
unzip yalefaces.zip > /dev/null

# clean up files, image names
rm -rf __MACOSX
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
