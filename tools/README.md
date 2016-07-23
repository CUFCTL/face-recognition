# Tools

This foler contains useful tools for performing face detection, facial recognition, and setting up directories to be tested on. Feel free to contribute any tools which you find useful for running the FCT Facial Recognition software.

## FaceCrop

FaceCrop is a tool that takes an input directory containing subdirectories of different subjects. FaceCrop will iterate through each subdirectory and crop all the faces in the image, then save the cropped faces in a subdirectory of a specified output directory. The subdirectories in the output directory will have the same names. This tool can be used to significantly expedite the process of creating images that are acceptable to run through the face detection system. Additionally, faces cropped to these dimensions could increase the accuracy of the system by having less 'noise' near the borders of the images.

FaceCrop usage:

    'python crop.py [path/to/source/directory] [path/to/output/directory]'

FaceCrop will create a directory in the pwd called cropped_test_set that contains the subdirectories of all the cropped
faces from the source directory.

## OpenCV
To run FaceCrop, you must first download openCV on your machine. To download openCV, go to [this site](http://opencv.org/downloads.html) and select the appropriate file to download. Then, run the following commands in the directory openCV that gets created when you unzip the download:

    sudo apt-get install cmake
    sudo apt-get install libgtk2.0-dev
    sudo apt-get install python-numpy

Make sure you are in the directory created by unzipping the downlaoded openCV package and then:

    cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_NEW_PYTHON_SUPPORT=ON -D CMAKE_INSTALL_PREFIX=/usr/local ./

You should see "Configuring done." and "Generating done." at the end of the make. Then:

    make

And finally:

    sudo make install

If you have any issues with the install, check out [this link](https://www.youtube.com/watch?v=MqQB5KKJCh0) for additional help.

## How To Run

Once you have openCV installed, running FaceCrop should not be a problem. Run "make" and then ./facedetect to crop photos. Currently the path to the photo is hard coded within facedetect.cpp, but future versions will create command line arguments and possible scripts to run this program on an entire directory.

## Issues

This is the 'first cut' at the FaceCrop tool, so do not be too critical! FaceCrop is being tested, but there are definitely some bugs at the moment to be worked out. Updates to come.
