#lda

LDA code for Biometric Creative Inquiry

##File descriptions


###Core components:

####example:
- Top-level executable / driver
- Runs CreateDatabase, FisherfaceCore, and Recognition in that order
- Defines relative train paths and test paths; change these manually

####CreateDatabase:
- Aligns a set of face images into a single 2D matrix
- Outputs a matrix where each column is a linearized image
- Defines and implements the database_t datatype
- #defines WIDTH and HEIGHT; change these if necessary when changing image paths

####FisherfaceCore:
- Converts image database and projects into facespace
- Images of the same person move closer together in the facespace and vice versa
- Most computation is done through heavy use of matrix arithmetic

####Recognition:
- Compares two faces by projecting the images into facespace and measures the Euclidean distance between them.

###Datatypes and auxiliary

####matrix:
- Defines and implements functions for our MATRIX datatype
- Allocates memory contiguously for compatibility with CBLAS and LAPACKE
  libraries
- Lncludes constructor, destructor, print function, mean function

####grayscale:
- Converts a PPM-format image to grayscale

####ppm:
- Contains all functions dealing with a PPM image which include - constructor, destructor, read header, and convert from P3 -> P6 (changing the magic number)
