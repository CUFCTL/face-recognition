# Tools

This folder contains tools to be used around the face recognition system. Feel free to add tools as you think of them.

## Dependencies

#### OpenCV

OpenCV is a computer vision library that has interfaces in C++ and Python. The easiest way to install OpenCV is to install the python package:

    # with apt-get
    sudo apt-get install python-opencv

    # or with pip
    pip install opencv-python

## FaceCrop

FaceCrop is a tool that takes an image, detects each face in the image, and saves each face as a cropped image. This tool can be used to significantly expedite the process of creating images that are acceptable to run through the face recognition system. Cropping images in this manner should reduce the amount of noise around the face and increase the accuracy of recognition.

FaceCrop usage:

    python crop.py [input-image]

FaceCrop saves the cropped images in a new directory called `[input-image]_cropped`.
