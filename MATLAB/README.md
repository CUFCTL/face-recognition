# face-recognition/archive

This folder contains the original Matlab code for the three algorithms in the project. The algorithms are described below, according to the documentation available in the project's Box folder.

## PCA

## LDA

The LDA algorithm is based on the PCA algorithm. It trains the same database as PCA, then performs a projection of that database based on the matrix's _singular values_, which are the _eigenvalues_ of the matrix times its transpose, that is the SV's of the matrix __A__ are the eigenvalues of the matrix __A__ * __A'__.

The files in the LDA folder are:

 - CreateDatabase.m
 - FisherfaceCore.m
 - Recognition.m
 - OutputData_binary.m
 - OutputData.m
 - example.m

The LDA Algorithm performs the following steps:

 - CreateDatabase.m
  - First, a matrix is constructed with the each column equal to one of the images from the training set.
 - FisherfaceCore.m
  - Next, a difference matrix is made by subtracting each image from the mean image.
  - A covariance matrix of the difference matrix is made by multiplying __A__ * __A'__.
  - The eigenvalues of this covariance matrix are sorted, with the least significant being eliminated
  - Each difference image is projected onto the new eigenspace, and the result is concatenated onto a matrix of the projected images.
  - Finally, the scatter matrices within and between each class are computed for the matrix of projected images. The generalized eigenvalues of these two matrices are ??? (What are they).
 - Recognition.m

## ICA

ICA is now being implemented by FastICA from http://research.ics.aalto.fi/ica/fastica/
