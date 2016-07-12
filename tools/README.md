# Tools 

This foler contains useful tools for performing face detection, facial recognition, and setting up directories to be tested on. Feel free to contribute any tools which you find useful to running the FCT Facial Recognition software.

## FaceCrop

FaceCrop is a tool that takes an input image (.jpg, .png, .bmp, etc) and detects all faces within the image. Once detected, the tool crops the image to minimum dimensions that contain the face. FaceCrop will allow for much more freedom when it comes to creating test sets. Now it is not necessary to require a picture to be taken at a precise distance and with one subject. 

### OpenCV
To run FaceCrop, you must first download openCV on your machine. To download openCV, go to [this](http://opencv.org/downloads.html) site and select the appropriate file to download. Then, run the following commands in the directory openCV that gets created when you unzip the download:

sudo apt-get install cmake (if you dont have cmake)
sudo apt-get install libgtk2.0-dev
sudo apt-get install python-numpy (this will be good to have if we utilize python scripts in the future)

Make sure you are in the directory created by unzipping the downlaoded openCV package and then:

cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_NEW_PYTHON_SUPPORT=ON -D CMAKE_INSTALL_PREFIX=/usr/local ./

You should see "Configuring done." and "Generating done." at the end of the make. Then:

make

And finally:

sudo make install

If you have any issues with the install, check out [this link](https://www.youtube.com/watch?v=MqQB5KKJCh0) for additional help.

#### How To Run

Once you have openCV installed, running FaceCrop should not be a problem. Run "make" and then ./facedetect to crop photos. Currently the path to the photo is hard coded within facedetect.cpp, but future versions will create command line arguments and possible scripts to run this program on an entire directory. 

### Issues

This is the 'first cut' at the FaceCrop tool, so do not be too critical! FaceCrop is being tested, but there are definitely some bugs at the moment to be worked out. Updates to come.
