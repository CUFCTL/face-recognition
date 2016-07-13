% script Arch2.m
% Finds ICA representation of train and test images under Architecture II, 
% described in Bartlett & Sejnowski (1997, 1998), and Bartlett, Movellan & 
% Sejnowski (2002):  In Architecture II, we load N principal component coefficients
% into rows of x, and then run ICA on x.
%
% Put aligned training images in the rows of C, one image per row.  
% In the following examples, there are 500 images of aligned faces of size 
% 60x60 pixels, so C is 500x3600. 
%
% You can use the following matlab code to create C:
% markFeatures.m collects eye and mouth positions. 
% align_Faces.m crops, aligns, and scales the face images.
% loadFaceMat.m loads the images into the rows of C. 
%
% This script also calls the matrix of PCA eigenvectors organized in 
% the columns of V (3600x499), created by [V,R,E] = pcabigFn(C');
%
% The ICA representation will be in F (called U in Bartlett, Movellan & 
% Sejnowski, 2002): 

loadFaceMat
loadTestMat

[V,R,E] = pcabigFn(C');
%D = zeroMn(C')'; % D is 500x3600 and D = C-ones(500,1)*mean(C);
%R = D*V; 	 % R is 500x499 and contains the PCA coefficients;

x = R(:,1:116)'; 	% x is 200x500;
runica 			% calculates w, wz, and uu. The matrix x gets overwritten
			% by a sphered version of x. 
F = uu'; 		% F is 500x200 and each row contains the ICA2 rep of 1 image. 
			% F = w * wz * zeroMn(R(:,1:200)')'; is the same thing.

% Representations of test images under architecture II
% Put original aligned test images in rows of Ctest:
%Ctest = []
%[FName, PName, FIndex] = uigetfile();
%I = imread(strcat(PName, FName));
%tmp = mat2gray(double(I));
%tmp = reshape(tmp,1,size(tmp,1)*size(tmp,2));
%Ctest = [Ctest;tmp];
Dtest = zeroMn(Ctest')'; % For proper testing, subtract the mean of the 
			 % training images not the test images: 
			 % Dtest = Ctest-ones(500,1)*mean(C);
Rtest = Dtest*V;
Ftest = w * wz * zeroMn(Rtest(:,1:116)');

% Test nearest neighbor classification using cosine, not euclidean distance, 
% as similarity measure.
%
% First create label vectors. These are column vectors of integers. Lets 
% say our 500 training examples consisted of 500 different people. Then
trainClass = [1:400]'; 
%
% We also need the correct class labels of the test examples if we want to 
% compute percent correct. Lets say the test examples were two images each 
% of the first 10 individuals. Then 
testClass = [1:20]';

%We now compute percent correct:
train_ex = F';
test_ex = Ftest;
[pc,rankmat] = nnclassFn(train_ex,test_ex,trainClass,testClass);

%pc is percent correct of first nearest neighbor.
%rankmat gives the top 30 matches for each test image. 

