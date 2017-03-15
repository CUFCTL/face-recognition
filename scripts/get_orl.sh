#!/bin/bash
# Download and extract the ORL face dataset.

mkdir -p datasets
cd datasets

# download archive if necessary
if [ ! -f att_faces.tar.Z ]; then
	wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z
fi

# extract archive
rm -rf orl_faces
tar -xzf att_faces.tar.Z
