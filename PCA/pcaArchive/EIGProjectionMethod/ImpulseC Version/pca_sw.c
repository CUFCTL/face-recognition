#include "co.h"
#include "cosim_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ppm.h"

#define eigen_type double // types of values for eigenfaces and projected images matrices

#define DEBUG 0 // 1 for printf debug statements
#define PROFILING 0 // 1 if testing recognition time

int imgloop = 3; // number of images to loop through for profiling - max of 10 for now

long int images, imgsize; // number of images and their size
clock_t start, end; // profiling clocks
clock_t startfpga, endfpga;

eigen_type **eigenfacesT, **projectedimages; // 2D matrices for eigenfaces and projected images read in from MATLAB file
int *mean; // 1D matrix of average pixel values from training database also read in from MATLAB file

// fill the three matrices above with data from the MATLAB file
void MatrixRead(char *filename)
{
   long int i, j;
   eigen_type temp;
   long int sizeW, sizeH;

   FILE *f = fopen(filename, "r");
   if (f == NULL) { printf("Failed to open eigenfaces file: %s\n", filename); return; }
   
   // first line of file contains the number of images and their size in pixels
   fscanf(f, "%ld", &images);
   fscanf(f, "%ld", &imgsize);
   
   // read eigenfaces
   sizeW = imgsize;
   sizeH = images;
   
   eigenfacesT = (eigen_type **)malloc(sizeH*sizeof(eigen_type *));
   for(i = 0; i<sizeH; i++)
   {
         eigenfacesT[i] = (eigen_type *)malloc(sizeW*sizeof(eigen_type));
   }

   j = 0; // row
   i = 0; // column

   while(!feof(f))
   {
      if(i >= sizeW)
      {
        if(++j == sizeH) break;
         i = 0;
      }

      fscanf(f, "%le", &temp);
      eigenfacesT[j][i++] = temp;
   }

   // read projected images
   sizeW = images;

   projectedimages = (eigen_type **)malloc(sizeH*sizeof(eigen_type *));
   for(i = 0; i<sizeH; i++)
   {
         projectedimages[i] = (eigen_type *)malloc(sizeW*sizeof(eigen_type));
   }

   j = 0; 
   i = 0;

   while(!feof(f))
   {
      if(i >= sizeW)
      {
        if(++j == sizeH) break;
         i = 0;
      }

      fscanf(f, "%le", &temp);
      projectedimages[j][i++] = temp;
   }
 
  // read mean 
   sizeW = imgsize;
   sizeH = 1;
   
   mean = (int *)malloc(sizeW*sizeof(int));

   j = 0; 
   i = 0; 

   while(!feof(f))
   {
      if(i >= sizeW) break;

      fscanf(f, "%d", &mean[i++]);
   }

   fclose(f);  
}

void Producer(co_stream output_stream, co_parameter iparam)
{
	cosim_logwindow log;
	long int row, col;

	co_int32 i, j, k, testnum;
	int testloop[] = {2,3,19,29,70,93,108,146,182,268}; // test image numbers present in ImpulsePCA directory (ppm image files)
	char inputimage[10];

	// create the Array from the Eigenvector file

	// there are several different database sizes (look in ImpulsePCA directory for valid numbers [3,10,29,etc.])
	//printf("Enter size eigenfaces database to use: ");
	//scanf("%d", &i);
	i = 4;

	// automatically runs through this many recognitions
	//printf("Enter number of faces to test: ");
	//scanf("%d", &imgloop);
	imgloop = 2;

	char eigendatabase[15];
	sprintf(eigendatabase, "eigenfaces%d.txt", i);

	MatrixRead(eigendatabase);

	if(PROFILING) start = clock();

	for(k = 0; k < imgloop; k++)
	{
		sprintf(inputimage, "%d.ppm", testloop[k]);

		// read in test image and convert it to grayscale
		PPMImage* testimage = ppm_image_constructor(inputimage);
		grayscale(testimage);

		// normalize input image by subtracting database mean
		for(i = 0; i < imgsize; i++)
		{
			testimage->pixels[i].intensity = testimage->pixels[i].intensity - mean[i] + 1;

			if(DEBUG && i < 5) printf("testimage->pixels[%d].intensity=%d (mean was %d)\n", i, testimage->pixels[i].intensity, mean[i]);
		}

		// project test image (ProjectedTestImage = eigenfacesT * NormalizedInputImage)
		double projectedtestimage[images];
		for(i = 0; i < images; i++)
		{
			projectedtestimage[i] = 0;
			for(j = 0; j < imgsize; j++)
			{
				projectedtestimage[i] += eigenfacesT[i][j] * testimage->pixels[j].intensity;
				//if(DEBUG && j < 5) printf("i=%d j=%d eigenfacesTij=%le intensityij=%d\n", i, j, eigenfacesT[i][j], testimage->pixels[j].intensity);
			}
			//if(DEBUG) printf("projectedtestimage[%d]=%lf\n", i, projectedtestimage[i]);
		}
	
		int min_index;
		double difference, distance, smallest = 1e20;

		log = cosim_logwindow_create("Producer");

		cosim_logwindow_write(log, "Process Producer opening stream: output_stream\n");
		co_stream_open(output_stream, O_WRONLY, INT_TYPE(32));

		// send data to FPGA through output_stream
		// (for now, send image number to test data passage to consumer)
		//co_stream_write(output_stream, &k, sizeof(int));

		for(i = 0; i < images; i++)
		{
			// find distance compared with projectedimages[i]
			distance = 0;

			for(j = 0; j < images; j++)
			{
				difference = projectedimages[j][i] - projectedtestimage[j];
				distance = distance + difference * difference;
				if(DEBUG) printf("image[%d] projimg=%e projtestimg=%e", i, projectedimages[j][i], projectedtestimage[j]);
				if(DEBUG) printf("\n difference = %e  distance is now %e\n",difference, distance);
			}

			if(DEBUG) printf("\n");

			if(DEBUG & !PROFILING) printf("distance[%d] = %e\n====\n", i, distance);

			if(distance < smallest)
			{
				smallest = distance;
				min_index = i;

			}
		}

		if(!PROFILING) printf("Matched image index is %d\n\n", min_index);
	}

	if(PROFILING)
	{
		end = clock();

		int runtime = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;

		printf("%d image recognitions with database size of %d took %dms", imgloop, images, runtime);
	}

	cosim_logwindow_write(log, "Process Producer closing stream output_stream\n");
	co_stream_close(output_stream);
  
	cosim_logwindow_write(log, "Process Producer exiting\n");

	scanf("%d", &testnum);
}

void Consumer(co_stream input_stream)
{
	int s;

	cosim_logwindow log;
	log = cosim_logwindow_create("Consumer");
	cosim_logwindow_write(log, "Process Consumer entered\n");

	cosim_logwindow_write(log, "Process Consumer opening stream: input_stream\n");
	co_stream_open(input_stream, O_RDONLY, INT_TYPE(32));

	// read data from FPGA through input_stream
	while (co_stream_read(input_stream, &s, sizeof(int)) == co_err_none) {
		cosim_logwindow_fwrite(log, "Process Consumer read %d from stream: input_stream\n", s);
		printf("Consumer read %d from stream: input_stream\n", s);
	}

	cosim_logwindow_fwrite(log, "Process Consumer closing stream input_stream\n");
	co_stream_close(input_stream);

	cosim_logwindow_fwrite(log, "Process Consumer exiting\n");
}
