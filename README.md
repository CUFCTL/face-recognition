# Face Recognition

This repository contains the code for the face recognition system developed by the FACE creative inquiry. We are developing an accelerated, real-time recognition system based on several popular face recognition techniques.

## Installation

Ubuntu (or some other Linux distribution) is the easiest and most recommended OS to use for this project. If you don't have a a Linux distribution, you have a few options: (1) set up a dual-boot, (2) set up a virtual machine, or (3) use a remote Linux machine (such as the Palmetto cluster) through SSH.

Once you have installed all dependencies listed below, you can build this project with `make`.

On Ubuntu, you can install several dependencies through `apt-get`:
```
sudo apt-get install cmake gcc gfortran git libopencv-dev
```

On Palmetto, these dependencies are available as modules:
```
module add cmake/3.6.1 cuda-toolkit/8.0.44 gcc/4.8.1 git opencv/2.4.9
```

### mlearn

[mlearn](https://github.com/CUFCTL/mlearn) is a machine learning library that was spawned from this project. Clone this repo and follow the instructions in the README to build the library.

### OpenCV for Python

If you want to use the `face-crop` tool, you will need to install the Python package for OpenCV:
```
sudo apt-get install python-opencv
```

## Usage

This repository includes a `face-rec` executable, which performs training, testing, and real-time recognition, and several helper scripts for getting datasets, running experiments, and face cropping.

The examples below use the ORL database, which you can set up with `./scripts/get_orl.sh`.

## Training and Testing

Run `./face-rec` without any arguments to view all of the options.

To train and test the system with PCA and a 70/30 dataset partition:
```
./scripts/create-sets.py -d orl -t 70 -r 30
./face-rec --train train_data --test test_data --feat pca --loglevel=3
```

## Cross Validation

To train and test the system 5 items with PCA and 70/30 dataset partitions:
```
./scripts/cross-validate.py -d orl -t 70 -r 30 -i 5 -- --pca
```

Notice that arguments for `./face-rec` must be placed after a `--`. You can supply any valid arguments for `./face-rec in this way:
```
./scripts/cross-validate.py -d orl -t 70 -r 30 -i 5 -- --pca --pca_n1=20 --loglevel=3
```

## Real-time Face Recognition

To create a database and perform real-time recognition on a video stream:
```
./scripts/create-sets.py -d orl -t 100
./face-rec --train train_data --feat pca
./face-rec --stream --feat pca
```

`face-rec` will use the default video stream and perform face detection and recognition on each video frame in real time.
