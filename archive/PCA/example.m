% A sample script, which shows the usage of functions, included in
% PCA-based face recognition system (Eigenface method)
%
% See also: CREATEDATABASE, EIGENFACECORE, RECOGNITION
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
clear
clc
close all

format long e

% TODO: set train and test path with arguments
TrainDatabasePath = '../../orl_faces_ppm/';
TestDatabasePath = '../../orl_faces_ppm/';

TrainFiles = dir(TrainDatabasePath);

TestImage = strcat(TestDatabasePath, '/s1_1.ppm');

% create training database
T = CreateDatabase(TrainDatabasePath);

[m, A, Eigenfaces] = EigenfaceCore(T);
[numPixels, numImages] = size(T);
[~, numFaces] = size(Eigenfaces);

save eigenfaces_200.txt numImages numFaces numPixels Eigenfaces A m -ascii

% perform recognition algorithm
OutputName = Recognition(TestImage, m, A, Eigenfaces);

SelectedImage = strcat(TrainDatabasePath,'/',TrainFiles(OutputName).name);

% display test image and matched image
im1 = imread(TestImage);
im2 = imread(SelectedImage);

image(im1)
title('Test Image');
figure, image(im2);
title('Equivalent Image');

fprintf('Test image is : %s\n', TestImage);
fprintf('Matched image is : %s\n', TrainFiles(OutputName).name);
