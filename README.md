# Facial Recognition Creative Inquiry

This repository contains the code for the face recognition system developed by the FACE creative inquiry. We are developing an accelerated, real-time recognition system based on several popular face recognition techniques.

## Getting Started

New team members should look at the [Wiki](https://github.com/CUFCTL/face-recognition/wiki), especially the pages on Git and Installation.

## Testing

To run an automated test (Monte Carlo cross-validation) with the ORL face database:

    # test with 3 random samples removed, 10 iterations
    ./scripts/cross-validate.sh -p orl_faces --c-only -t 3 -i 10 [--pca --lda --ica]

To test MATLAB code with ORL database:

    # test with 3 random samples removed, 10 iterations
    ./scripts/cross-validate.sh -p orl_faces --matlab-only -t 3 -i 10 [--pca --lda --ica]

To generate profiling information with `gprof`:

    ./face-rec ...
    gprof ./face-rec > [logfile]

## The Image Library

This software currently supports a subset of the [Netpbm](https://en.wikipedia.org/wiki/Netpbm_format) format, particularly with PGM and PPM images.

Images should __not__ be stored in this repository! Instead, images should be downloaded separately. Face databases are widely available on the Internet, such as [here](http://web.mit.edu/emeyers/www/face_databases.html) and [here](http://face-rec.org/databases/). I am currently using the [ORL database](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html):

    wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z
    tar -xvzf att_faces.tar.Z
    rm orl_faces/README

To convert JPEG images to PGM with ImageMagick:

    ./scripts/convert-images.sh [src-folder] [dst-folder] jpeg pgm

## Results

Not quite ready
