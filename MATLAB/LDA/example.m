% A sample script, which shows the usage of functions, included in
% FLD-based face recognition system (Fisherface method)
%
% See also: CREATEDATABASE, FISHERFACECORE, RECOGNITION
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
% load_stuff: flag
%
%
% Good Reference? : http://www.eecs.umich.edu/~silvio/teaching/lectures/lecture22.pdf
%
clear
clc
close all

TrainDatabasePath = '../../train_images_ppm/';
TestDatabasePath = '../../test_images_ppm/';
Class_number = 40;

T = CreateDatabase(TrainDatabasePath);
[m, V_PCA, V_Fisher, ProjectedImages_Fisher] = FisherfaceCore(T, Class_number);

TrainFiles = dir(strcat(TrainDatabasePath, '/*.ppm'));
TestFiles = dir(strcat(TestDatabasePath, '/*.ppm'));

for i = 1 : size(TestFiles, 1)
    % perform recognition algorithm
    strtest = strcat(TestDatabasePath, '/', TestFiles(i).name);
    j = Recognition(strtest, m, V_PCA, V_Fisher, ProjectedImages_Fisher);

    % print results
    fprintf('test image: \"%s\"\n', TestFiles(i).name);
    fprintf('       LDA: \"%s\"\n', TrainFiles(j).name);
    fprintf('\n');
end
