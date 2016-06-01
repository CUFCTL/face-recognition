README.txt

This file contains important information on the implementation and execution of the executable PCA.

PCA requires a txt file named "eigefaces_200.txt." This file can be created by running example.m (this code does not run all the way through but will run far enough to create the necessary file.

This file contains important information about the image database.
	It contains

	the number of images represented by the file
	the number of faces represented by the images
	the number of pixels per images
	Eigenfaces             - (M*Nx(P-1)) Eigen vectors of the covariance matrix of the training database
        A                      - (M*NxP) Matrix of centered image vectors
	m                      - (M*Nx1) Mean of the training database
        

Here is the program flow for PCA.

First, Main calls LoadTrainingDatabase which does the following:
	it opens a file (hard coded in Main to be called "eigenfaces_200.txt."
	it then reads several values into other variables passed into LoadTrainingDatabase.
		images - the number of images
		facessize - number of faces (should be the same as number of images)
		imgsize - size of the images in pixels
	then it reads several arrays from the file (listed above and and again below)
		eigenfacesT - eigenvectors of the covariance matrix of the training database
		projectedimages - matrix of centered image vectors
		mean - mean of the training database
	then the function returns to Main.

Next, Main asks the user for the number of desired loops for the testing function "Recognition"

Main then calls calculate_projected_images which does the following: 
	it multiplies all of the aligned images with all of the eigenvectors of the covariance matrix.

	so the aligned images array (projectedimages) is imgsize X images
	and the eigenvector matrix (eigenfacesT) is imgsize X num_faces so
	projectedtrainimages = eigenfacesT' * projectedimages

	the function then returns to Main.

Main then the Recognition function 'n' times as designated by the user, and it does the following:
	begins by opening a .ppm image designated by the value in testloop (in pca.c). It converts the designated image to grayscale. It then "normalizes" the image by
	subtracting the database average from the image. It then creates a projected test image by multiplying the normalized image to the eigenvectors (same process as for the 
	projectedtrainimage in pca.c). Finally, it calculates the "distance" of the projected test image with the projected train images and finds the training image with the smallest 
	distance from the test image. It prints the result and returns to Main.

Main finishes by printing a message of termination and frees all the necessary pointers.