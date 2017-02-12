#!/bin/bash
# Download and extract the ORL face database.

mkdir -p datasets
cd datasets

wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z
tar -xzf att_faces.tar.Z
