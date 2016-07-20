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

TrainDatabasePath = '../../train_images_ppm/';
TestDatabasePath = '../../test_images_ppm/';

TrainFiles = dir(strcat(TrainDatabasePath, '/*.ppm'));
TestFiles = dir(strcat(TestDatabasePath, '/*.ppm'));

% create training database
T = CreateDatabase(TrainDatabasePath);

[m, A, Eigenfaces] = EigenfaceCore(T);
[numPixels, numImages] = size(T);
[~, numFaces] = size(Eigenfaces);

ProjectedImages = Eigenfaces' * A;

% test each image in the test set
for i = 1 : size(TestFiles, 1)
    % perform recognition algorithm
    strtest = strcat(TestDatabasePath, '/', TestFiles(i).name);
    j = Recognition(strtest, m, Eigenfaces, ProjectedImages);

    % print results
    fprintf('test image: \"%s\"\n', TestFiles(i).name);
    fprintf('       PCA: \"%s\"\n', TrainFiles(j).name);
    fprintf('\n');
end
