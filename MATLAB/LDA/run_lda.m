% A sample script, which shows the usage of functions, included in
% FLD-based face recognition system (Fisherface method)
%
% See also: CREATEDATABASE, FISHERFACECORE, RECOGNITION
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
%
% Good Reference? : http://www.eecs.umich.edu/~silvio/teaching/lectures/lecture22.pdf
%
clear
clc
close all

TrainDatabasePath = '../../train_images/';
TestDatabasePath = '../../test_images/';

% create training database
[TrainFiles, T, Class_number] = CreateDatabase(TrainDatabasePath);
[m, V_PCA, V_Fisher, ProjectedImages_Fisher] = FisherfaceCore(T, Class_number);

% test each image in the test set
TestFiles = dir(strcat(TestDatabasePath, '/*.pgm'));
num_correct = 0;

for i = 1 : size(TestFiles, 1)
    % perform recognition algorithm
    strtest = strcat(TestDatabasePath, '/', TestFiles(i).name);
    j = Recognition(strtest, m, V_PCA, V_Fisher, ProjectedImages_Fisher);

    % print results
    fprintf('test image: \"%s\"\n', TestFiles(i).name);
    fprintf('       LDA: \"%s/%s\"\n', TrainFiles(j).class, TrainFiles(j).name);
    fprintf('\n');

    % determine whether the algorithm was correct
    % assumes that filename is formatted as '{class}_{index}.ppm'
    tokens_test = strsplit(TestFiles(i).name, '_');

    if strcmp(TrainFiles(j).class, tokens_test{1})
        num_correct = num_correct + 1;
    end
end

success_rate = num_correct / size(TestFiles, 1) * 100;

fprintf('%d / %d matched, %.2f%%\n', num_correct, size(TestFiles, 1), success_rate);
