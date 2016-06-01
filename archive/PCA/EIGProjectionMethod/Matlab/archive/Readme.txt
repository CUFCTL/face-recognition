'Eigenface' Face Recognition System
Written by: Amir Hossein Omidvarnia
Email: aomidvar@ece.ut.ac.ir           

This package implements a well-known PCA-based face recognition 
method, which is called 'Eigenface' [1].
All functions are easy to use, as they are heavy commented. 
Furthermore, a sample script is included to show their usage. 
In general, you should follow this order:

1. Select training and test database paths.
2. Select path of the test image.
3. Run 'CreateDatabase' function to create 2D matrix of all training images.
4. Run 'EigenfaceCore' function to produce basis's of facespace.
5. Run 'Recognition' function to get the name of equivalent image in training database.  

For your convenience, I have prepared sample training and test databases, which are parts 
of 'face94' Essex face database [2]. You just need to copy the above functions, along with 
the training and test databases into a specified path (for example 'work' path of your
MATLAB root). Then follow dialog boxes, which will appear upon running 'example.m'.

Enjoy it!



References:

[1] P. N. Belhumeur, J. Hespanha, and D. J. Kriegman. Eigenfaces vs. Fisherfaces: Recognition 
    using class specific linear projection. In ECCV (1), pages 45--58, 1996.

[2] Available at:
    http://cswww.essex.ac.uk/mv/allfaces/faces94.zip