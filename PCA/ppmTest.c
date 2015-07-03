#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main (void) {
	

	matrix_t *PPM = initializeMatrix (UNDEFINED, 92 * 112, 1);
	unsigned char *pixels = malloc (3 * 92*112*sizeof(unsigned char));
	loadPPMtoMatrixCol (".//images//1.ppm", PPM, 0, pixels);
	
	writePPMgrayscale (".//wahaha.ppm", PPM, 0, 92, 112);

	freeMatrix (PPM);
	free (pixels);
	return 0;
}
