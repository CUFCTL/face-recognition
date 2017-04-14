#!/bin/bash
# Extract the FERET dataset.
#
# Since the FERET dataset cannot be downloaded
# directly from the Internet, this script assumes that
# the archive has already been downloaded into the datasets
# directory.
#
# The FERET dataset contains several resolutions of the images,
# which are separated by folders:
#
#   images/      (512 x 768)
#   smaller/     (256 x 384)
#   thumbnails/  (128 x 192)
#
# This script can be modified to use any resolution.

SIZE="thumbnails"

mkdir -p datasets
cd datasets

# extract archive
if [ ! -d colorferet ]; then
	if [ ! -f colorferet.tar ]; then
		>&2 echo "error: datasets/colorferet.tar does not exist"
		exit
	fi

	echo "Extracting archive..."
	tar -xf colorferet.tar
fi

# extract images from archive
echo "Extracting images from archive (SIZE=$SIZE)"

rm -rf feret
mkdir feret

cp -r colorferet/dvd1/data/$SIZE/* feret
cp -r colorferet/dvd2/data/$SIZE/* feret

chmod 755 -R feret

bunzip2 feret/**/*.ppm.bz2
