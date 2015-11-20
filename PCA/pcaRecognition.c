/* Things to make faster that are not with cuda (or maybe in cuda and I don't think it can be)
	- I did all of these
*/

/* Things to cuda-ize (technical term)
	- matrix multiply
	- parts of reading in the PPM
	- vector subtract
	- distance formula
	- finding the min
	- any other loops
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <float.h>

#include "matrixOperations.h"


int main (int argc, char **argv) {

	FILE * projectedImagesFile = fopen ("testDB.dat", "r");
	FILE * filenamesFile = fopen ("filenamesDB.dat", "r");
	FILE * testImagesFile = fopen (argv[1], "r");

	int i, j;
	//char databasePath[] = ".//images//";
	char testPath[] = ".//test_images//";
	char *path = malloc (200 * sizeof (char));
	char *testImagePath = malloc (200 * sizeof (char));

	// Read in the projected images, eigenfaces, and the mean face
	/*matrix_t *projectedImages = fscanMatrix (projectedImagesFile);
	matrix_t *eigenfaces = fscanMatrix (projectedImagesFile);
	matrix_t *m = fscanMatrix (projectedImagesFile); */

	matrix_t *projectedImages = m_fread (projectedImagesFile);					//freadMatrix to m_fread
	matrix_t *transposedEigenfaces = m_fread (projectedImagesFile);				//
	matrix_t *m = m_fread (projectedImagesFile);								//


	// Read in the filenames
	char **filenames = malloc (projectedImages->numCols * sizeof (char *));
	for (i = 0; i < projectedImages->numCols; i++) {
		filenames[i] = malloc (200 * sizeof (char));
		fscanf (filenamesFile, "%s", filenames[i]);
	}

	// Recognize all images specified in the file
	unsigned char *pixels = (unsigned char *) malloc (3 * m->numRows * sizeof (unsigned char));
	while (fscanf(testImagesFile, "%s", testImagePath) && !feof (testImagesFile)) {
		// Read in a test image
		sprintf (path, "%s%s", testPath, testImagePath);
		matrix_t *testImage = m_initialize (UNDEFINED, m->numRows, 1);			// initializeMatrix to m_initialize

		loadPPMtoMatrixCol (path, testImage, 0, pixels);						// Already using shared library

		// Take the difference image
		subtractMatrixColumn (testImage, 0, m, 0);								// Needs shared library function
		matrix_t *differenceImage = testImage; // for naming standards

		// Project the image into the face space
		matrix_t *projectedTestImage = m_matrix_multiply (transposedEigenfaces, NOT_TRANSPOSED, differenceImage, NOT_TRANSPOSED, 0);		// matrixMultiply to m_matrix_multiply


		// Calculate the min Euclidean distance between the projectedTestImage and
		// the projectedImages
		// COMMENT : This is what I thought the matlab code was doing but I wasn't 100% sure so I would double check this
		double norm, temp;
		double min = DBL_MAX;
		int idx = -1;
		for (i = 0; i < projectedImages->numCols; i++) {
			norm = 0;
			for (j = 0; j < projectedImages->numRows; j++) {
                temp = elem(projectedTestImage, j, 0) - elem(projectedImages, j, i);
				norm += abs (temp);
			}
			norm *= norm;

			if (norm < min) {
				min = norm;
				idx = i;
			}
		}

		/* NOTE: if we wanted to set a threshold that the min would need to be
				under it would be here */
		/* COMMENT : It would be kind of neat to store the tested image and the
		recognized image and then after all the tests are over to display the images */
		printf ("tested image = %s\t", testImagePath);
		printf ("recognized image = %s\n", filenames[idx]);
	}

	return 0;
}
