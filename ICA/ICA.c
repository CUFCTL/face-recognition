#include <stdio.h>
#include "matrix_manipulation.h"

#define EIGV_NUM 116  // Can be changed for optimization
/*==================================================================================================
 *  ICA.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *  
 *  This file contains
 *      main
 *  
 *  Lasted Edited: Jun. 19, 2013
 *
 *  Changes made: by William - added pseudo code for Arch 1 and 2.
 *
 *  Description: Before this code can be run, several steps must be completed in Matlab. These 
 *  include running markFeatures.m, align_Faces.m, alignTestFaces.m, loadFaceMat.m, and 
 *  loadTestMat.m.
 *
 */
 
/*  essential libraries are included here */
void runica(data_t *matrix, data_t, int rows, int cols, data_t *uu, data_t *w, data_t *wz);

int main(int argc, char *argv[]) {

    /*=========================================Arch1========================================
    |   Description: 
    |
    |   THIS ARCHITECTURE CALLS
    |       pcabigFn
    |       runica
    |       zeroMn
    |       inv
    |       nnclassFn
    |
    |   COMMENTS FROM MATLAB CODE
    |       % script Arch1.m
    |       % Finds ICA representation of train and test images under Architecture I, 
    |       % described in Bartlett & Sejnowski (1997, 1998), and Bartlett, Movellan & 
    |       % Sejnowski (2002):  In Architecture I, we load N principal component 
    |       % eigenvectors into rows of x, and then run ICA on x.
    |       %
    |       % Put the aligned training images in the rows of C, one image per row.  
    |       % In the following examples, there are 500 images of aligned faces of size 
    |       % 60x60 pixels, so C is 500x3600. 
    |       %
    |       % You can use the following matlab code to create C:
    |       % markFeatures.m collects eye and mouth positions. 
    |       % align_Faces.m crops, aligns, and scales the face images.
    |       % loadFaceMat.m loads the images into the rows of C. 
    |       %
    |       % This script also calls the matrix of PCA eigenvectors organized in 
    |       % the columns of V (3600x499), created by [V,R,E] = pcabigFn(C');
    |       %
    |       % The ICA representation will be in the rows of F (called B in Bartlett, 
    |       % Movellan & Sejnowski, 2002):
    ======================================================================================*/
    data_t *uu;
	data_t *w;
	data_t *wz;
	data_t *V;
	data_t *R;
	data_t *E;
	data_t *C;
	data_t *length_matrix;
	data_t *x;
	data_t *oldx;
	data_t *Ctest;
	data_t *Dtest;
	data_t *Rtest;
	data_t *Ftest;
	data_t *trainClass;
	data_t *testClass;
	data_t *F;
	data_t *train_ex;
	data_t *test_ex;
	data_t *temp;
	data_t *temp2;
	
	data_t *temp_v;
	data_t *temp_R;
	data_t *trans_C;
	data_t *temp_Ctest;
	data_t *temp_Dtest;
	
	int num_images;
	int num_testimages;
	int num_pixels;
	int cols;

	////// Load in aligned PGM Face Images
	allocate_matrix(C, num_images, num_pixels);
	
	//Read_PGM_Folder
	
	allocate_matrix(V, num_pixels, num_images);
	allocate_matrix(R, num_images, num_images);
	allocate_matrix(E, 1, num_images);
	allocate_matrix(length_matrix, ?, ?);		// Still unsure of this size.
	
	allocate_matrix(x, EIGV_NUM, num_pixels);
	
	allocate_matrix(uu, EIGV_NUM, num_pixels);
	allocate_matrix(w, EIGV_NUM, EIGV_NUM);
	allocate_matrix(wz, EIGV_NUM, EIGV_NUM);
	
	allocate_matrix(F, num_images, EIGV_NUM);
	
	allocate_matrix(oldx, ?, ?);		// oldw in matlab?? 116x116 if so
	allocate_matrix(Ctest, num_testimages, num_pixels);		//
	allocate_matrix(Dtest, num_testimages, num_pixels);
	allocate_matrix(Rtest, num_testimages, num_images);
	allocate_matrix(Ftest, num_testimages, EIGV_NUM);
	
	allocate_matrix(trainClass, 1, 500);		// This is pre-transpose. Consider creating another array.
	allocate_matrix(testClass, 1, 20);			// Also pre-transpose.

	allocate_matrix(train_ex, EIGV_NUM, num_images);
	allocate_matrix(test_ex, EIGV_NUM, num_testimages);
	
	allocate_matrix(temp, EIGV_NUM, EIGV_NUM);
	allocate_matrix(temp2, EIGV_NUM, EIGV_NUM);
	allocate_matrix(trans_C, num_images, num_pixels);
	allocate_matrix(temp_v, num_pixels, EIGV_NUM);
	allocate_matrix(temp_R, num_images, EIGV_NUM);
	allocate_matrix(temp_Ctest, num_pixels, num_testimages);
	allocate_matrix(temp_Dtest, num_pixels, num_testimages);
	
	
	///// Begin ICA Operations
	
	transpose(trans_C, C, num_images, num_pixels);
	// length_matrix, B?
    pcabigFn(V, R, E, trans_C, num_pixels, num_images); 							//  [V,R,E] = pcabigFn(C');
																		//  %D = zeroMn(C')'; % D is 500x3600 and D = C-ones(500,1)*mean(C);
																		//  %R = D*V; 	 % R is 500x499 and contains the PCA coefficients;
																		//
																		//  % We choose to use the first 200 eigenvectors.
																		//  % (If PCA generalizes better by dropping first few eigenvectors, ICA will too).
    //FYI ALLL TRANSPOSES in this file MUST CHANGE LOOK AT MATRIX MANIPULATION
	
	submatrix(temp_v, V, num_pixels, num_images, 0, 0, num_pixels - 1, EIGV_NUM - 1);
    transpose(x, temp_v, num_pixels, EIGV_NUM);                                          //  x = V(:,1:116)'; 		% x is 200x3600
    runica(x, oldx, rows, cols, uu, w, wz);  //  runica 				  % calculates wz, w and uu. The matrix x gets
                                                    //                   % overwritten by a sphered version of x.
    multiply_matrices(temp, w, wz, EIGV_NUM, EIGV_NUM, EIGV_NUM);
	inv(temp2, temp, EIGV_NUM);
	//F = R(:,1:116) * inv(w*wz);                    					 //  F = R(:,1:116) * inv(w*wz); 	% F is 500x200 and each row contains the 
    //R(:,1:116);
	submatrix(temp_R, R, num_images, num_images, 0, 0, num_images - 1, EIGV_NUM - 1);
	multiply_matrices(F, temp_R, temp2, num_images, EIGV_NUM, EIGV_NUM);                                                //                                  % ICA1 rep of an image
																		//
																		//  % Representations of test images under architecture I:
																		//  % Put original aligned test images in rows of Ctest.
																		//  loadTestMat -   Will be done in Matlab
    //Dtest = transpose(zeroMn(transpose(Ctest)));    					//  Dtest = zeroMn(Ctest')'; % For proper testing, subtract the mean of the
	transpose(temp_Ctest, Ctest, num_testimages, num_pixels);
	zero_mean(temp_Dtest, temp_Ctest, num_pixels, num_testimages);
	transpose(Dtest, temp_Dtest, num_testimages, num_pixels);
	
																		//                           % training images not the test images:
																		//                           % Dtest = Ctest-ones(500,1)*mean(C);
    Rtest = Dtest * V;                              					//  Rtest = Dtest*V;
    //Ftest = Rtest(:,1:116) * inv(w * wz);         					  //  Ftest = Rtest(:,1:116) * inv(w*wz);
    Rtest(:,1:116);
	multiply_matrices(Ftest, Rtest, temp2, int rows, int cols, int k);                                                  //
                                                    //  % Test nearest neighbor classification using cosine, not euclidean distance, 
                                                    //  % as similarity measure.
                                                    //  %
                                                    //  % First create label vectors. These are column vectors of integers. Lets 
                                                    //  % say our 500 training examples consisted of 500 different people. Then
                                                    //  %trainClass = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8 8 8 9 9 9 9 10 10 10 10 11 11 11 11 12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15 16 16 16 16 17 17 17 17 18 18 18 18 19 19 19 19 20 20 20 20 21 21 21 21 22 22 22 22 23 23 23 23 24 24 24 24 25 25 25 25 26 26 26 26 27 27 27 27 28 28 28 28 29 29 29 29]'; 
    trainClass = transpose([1:500]);                //  trainClass = [1:500]';
                                                    //  %
                                                    //  % We also need the correct class labels of the test examples if we want to 
                                                    //  % compute percent correct. Lets say the test examples were two images each 
                                                    //  % of the first 10 individuals. Then 
                                                    //  %testClass = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]';
    testClass = transpose([1:20]);                  //  testClass = [1:20]';
                                                    //
    /*  Profiling begins here   */                  //  profile clear
                                                    //  profile -detail builtin on
                                                    //  %We now compute percent correct:
	transpose(train_ex, F, ?, ?);					//  train_ex = F';
	transpose(test_ex, Ftest, ?, ?);				//  test_ex = Ftest';
    nnclassFn(pc, rankmat, train_ex,test_ex,trainClass,testClass);  //  [pc,rankmat] = nnclassFn(train_ex,test_ex,trainClass,testClass)
    /*  End profiling   */                          //  profile off
                                                    //  profile viewer
                                                    //  %pc is percent correct of first nearest neighbor.
                                                    //  %rankmat gives the top 30 matches for each test image.

    /*=========================================Arch2========================================
    |   Description: 
    |
    |   THIS ARCHITECTURE CALLS
    |       pcabigFn
    |       runica
    |       zeroMn
    |       nnclassFn
    |
    |   COMMENTS FROM MATLAB CODE
    |       % script Arch2.m
    |       % Finds ICA representation of train and test images under Architecture II, 
    |       % described in Bartlett & Sejnowski (1997, 1998), and Bartlett, Movellan & 
    |       % Sejnowski (2002):  In Architecture II, we load N principal component coefficients
    |       % into rows of x, and then run ICA on x.
    |       %
    |       % Put aligned training images in the rows of C, one image per row.  
    |       % In the following examples, there are 500 images of aligned faces of size 
    |       % 60x60 pixels, so C is 500x3600. 
    |       %
    |       % You can use the following matlab code to create C:
    |       % markFeatures.m collects eye and mouth positions. 
    |       % align_Faces.m crops, aligns, and scales the face images.
    |       % loadFaceMat.m loads the images into the rows of C. 
    |       %
    |       % This script also calls the matrix of PCA eigenvectors organized in 
    |       % the columns of V (3600x499), created by [V,R,E] = pcabigFn(C');
    |       %
    |       % The ICA representation will be in F (called U in Bartlett, Movellan & 
    |       % Sejnowski, 2002): 
    ======================================================================================*/
	transpose(trans_C, C, ?, ?);
    pcabigFn(V, R, E, trans_C);    //  [V,R,E] = pcabigFn(C');
                                        //  %D = zeroMn(C')'; % D is 500x3600 and D = C-ones(500,1)*mean(C);
                                        //  %R = D*V; 	 % R is 500x499 and contains the PCA coefficients;
                                        //
    x = transpose(R(:,1:116));          //  x = R(:,1:116)'; 	% x is 200x500;
   runica(x, oldx, rows, cols, uu, w, wz);   //  runica 			    % calculates w, wz, and uu. The matrix x gets overwritten
			                            //                      % by a sphered version of x. 
    transpose(F, uu, ?, ?);             //  F = uu'; 		% F is 500x200 and each row contains the ICA2 rep of 1 image. 
			                            //                  % F = w * wz * zeroMn(R(:,1:200)')'; is the same thing.
                                        //
                                        //  % Representations of test images under architecture II
                                        //  % Put original aligned test images in rows of Ctest:
                                        //  %Ctest = []
                                        //  %[FName, PName, FIndex] = uigetfile();
                                        //  %I = imread(strcat(PName, FName));
                                        //  %tmp = mat2gray(double(I));
                                        //  %tmp = reshape(tmp,1,size(tmp,1)*size(tmp,2));
                                        //  %Ctest = [Ctest;tmp];
    Dtest = transpose(zeroMn(transpose(Ctest)));    //  Dtest = zeroMn(Ctest')'; % For proper testing, subtract the mean of the 
	    		                        //  % training images not the test images: 
		    	                        //  % Dtest = Ctest-ones(500,1)*mean(C);
    Rtest = Dtest * V;                  //  Rtest = Dtest*V;
    Ftest = w * wz * zeroMn(transpose(Rtest(:,1:116))); //  Ftest = w * wz * zeroMn(Rtest(:,1:116)');
                                        //
                                        //  % Test nearest neighbor classification using cosine, not euclidean distance, 
                                        //  % as similarity measure.
                                        //  %
                                        //  % First create label vectors. These are column vectors of integers. Lets 
                                        //  % say our 500 training examples consisted of 500 different people. Then
    trainClass = transpose([1:400]);    //  trainClass = [1:400]'; 
                                        //  %
                                        //  % We also need the correct class labels of the test examples if we want to 
                                        //  % compute percent correct. Lets say the test examples were two images each 
                                        //  % of the first 10 individuals. Then 
    testClass = transpose([1:20]);      //  testClass = [1:20]';
                                        //
                                        //  %We now compute percent correct:
    train_ex = transpose(F);            //  train_ex = F';
    test_ex = Ftest;                    //  test_ex = Ftest;
    nnclassFn(pc, rankmat, train_ex,test_ex,trainClass,testClass);  //  [pc,rankmat] = nnclassFn(train_ex,test_ex,trainClass,testClass);
                                        //
                                        //  %pc is percent correct of first nearest neighbor.
                                        //  %rankmat gives the top 30 matches for each test image. 

										
										
	free_matrix(&temp);
	free_matrix(&temp2);
    return 0;
}


/*==================================================================================================
void ICA(double v[], double r[], double e[]) {
 // Dtest = zeroMn(Ctest')'; For proper testing, subtract the mean of the training images not the test images: Dtest = Ctest-ones(500,1)*mean(C);
 // Rtest = Dtest*V;
 // Ftest = Rtest(:,1:116) * 1/(w*wz);

 // Test nearest neighbor classification using cosine, not euclidean distance, 
 // First create label vectors. These are column vectors of integers. Lets 
 // say our 500 training examples consisted of 500 different people. Then
 double trainClass[] = {1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8 8 8 9 9 9 9 10 10 10 10 11 11 11 11 12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15 16 16 16 16 17 17 17 17 18 18 18 18 19 19 19 19 20 20 20 20 21 21 21 21 22 22 22 22 23 23 23 23 24 24 24 24 25 25 25 25 26 26 26 26 27 27 27 27 28 28 28 28 29 29 29 29}; 

 // We also need the correct class labels of the test examples if we want to 
 // compute percent correct. Lets say the test examples were two images each 
 // of the first 10 individuals. Then 
 double testClass[] = {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29};

 // profile here
 // train_ex = F';
 // test_ex = Ftest';
 // [pc,rankmat] = nnclassFn(train_ex,test_ex,trainClass,testClass);
 
 
 // stop profiling    
}
*/


    /*=========================================runica========================================
    |   Description: 
    |   % Copyright 1996 Tony Bell
	|	% This may be copied for personal or academic use.
	|	% For commercial use, please contact Tony Bell 
	|	% (tony@salk.edu) for a commercial license.
	|
	|
	|	% Script to run ICA on a matrix of images. Original by Tony Bell. 
	|	% Modified by Marian Stewart Bartlett.
	|
	|	%Assumes image gravalues are in rows of x. Note x gets overwritten.
	|	%Will find N independent components, where N is the number of images.
	|
	|	%There must be at least 5 times as many examples (cols of x) as the
	|	%dimension of the data (rows of x). 
	|
    |   COMMENTS FROM MATLAB CODE
=======================================================*/
runica(data_t *matrix, data_t *oldx, int rows, int cols, data_t *uu, data_t *w, data_t *wz){
	data_t *xx;
	data_t *ID;
	data_t *oldw;
	data_t *olddelta;
	data_t *temp;
	data_t *temp2;
	
	/*  Start OF ARRAY ALLOCATION */
	allocate_matrix(xx, rows, cols);
	allocate_matrix(w, rows, rows);
	allocate_matrix(oldw, rows, rows);
	allocate_matrix(ID, rows, rows);
	allocate_matrix(olddelta, 1, rows*rows);
	allocate_matrix(temp, rows, cols);
	allocate_matrix(temp2, ?,?);
	allocate_matrix(temp3, ?,?);
	/*  END OF ARRAY ALLOCATION */
	
	int n = rows; int m = rows; int P = cols;														// N=size(x,1); P=size(x,2); M=N;	 %M is dimension of the ICA output
	spherex(matrix, oldx, rows, cols P, wz);														// spherex;                          % remove first and second order stats from x
	inv(temp2, wz, int rows);
																									// xx=inv(wz)*x;                     % xx thus holds orig. data, w. mean extracted.
	multiply_matrices(xx, temp2, x, int rows, int cols, int k);
	eye(w, rows);																					// %******** setup various variables
	int count = 0; int sweep = 0;																	// w=eye(N); count=0; perm=randperm(P); sweep=0; Id=eye(M);
	randperm(perm, P);																					// oldw=w; olddelta=ones(1,N*M); angle=1000; change=1000;
	eye(ID, rows);
	eye(oldw, rows); 																				// %******** Train. outputs a report every F presentations.
	ones(olddelta, 1, rows*rows);																	// % Watch "change" get small as it converges. Try annealing learning 
	double angle = 1000;																			// % rate, L, downwards to 0.0001 towards end.
	double change = 1000;																			// % For large numbers of rows in x (e.g. 200), you need to use a low 
																									// % learning rate (I used 0.0005). Reduce if the output blows 
																									// % up and becomes NAN. If you have fewer rows, use 0.001 or larger.

																									// B=50; L=0.0005; F=5000; for I=1:1000, sep96; end; 
																									// B=50; L=0.0003; F=5000; for I=1:200, sep96; end; 
																									// B=50; L=0.0002; F=5000; for I=1:200, sep96; end;   
	int I;																							// B=50; L=0.0001; F=5000; for I=1:200, sep96; end; 
    int B = 50;
    float L = 0.0005;
    int F = 5000; 

    for(I=1; I<1001; I++) {
        sep96(matrix, w, perm, sweep, count, N, M, P, B, L, angle, change, ID);
    }
 
    B = 50;
    L = 0.0003;
    F = 5000;

    for(I=1; I<201; I++) {
        sep96(matrix, w, perm, sweep, count, N, M, P, B, L, angle, change, ID);
    }

    B = 50;
    L = 0.0002;
    F = 5000;

    for(I=1; I<201; I++) {
        sep96(matrix, w, perm, sweep, count, N, M, P, B, L, angle, change, ID);
    } 

    B = 50;
    L = 0.0001;
    F = 5000;

    for(I=1; I<201; I++) {
        sep96(matrix, w, perm, sweep, count, N, M, P, B, L, angle, change, ID);
    }															

	// check rows and cols and k of the next lines
	multiply_matrices(temp, w, wz, rows, cols, int k);
	multiply_matrices(temp3, temp, xx, rows, cols, int k);			// %********
																	// uu=w*wz*xx;  % make separated output signals.
	covariance(uu, transpose(temp3), rows, cols);					// cov(uu')     % check output covariance. Should approximate 3.5*I.

	free_matrix(&temp);
	free_matrix(&temp2);
	free_matrix(&temp3);
}
