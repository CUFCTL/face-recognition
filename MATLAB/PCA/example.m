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
TrainDatabasePath = '../../train_images_ppm/';
TestDatabasePath = '../../test_images_ppm/';

TrainFiles = dir(strcat(TrainDatabasePath, '/*.ppm'));
TestFiles = dir(strcat(TestDatabasePath, '/*.ppm'));

% create training database
T = CreateDatabase(TrainDatabasePath);

[m, A, Eigenfaces] = EigenfaceCore(T);
[numPixels, numImages] = size(T);
[~, numFaces] = size(Eigenfaces);

% test each image in the test set
for i = 1 : size(TestFiles, 1)
    % perform recognition algorithm
    OutputName = Recognition(strcat(TestDatabasePath, '/', TestFiles(i).name), m, A, Eigenfaces);

    % print results
    fprintf('test image: \"%s\"\n', TestFiles(i).name);
    fprintf('\tPCA: \"%s\"\n', TrainFiles(OutputName).name);
    fprintf('\n');
end
