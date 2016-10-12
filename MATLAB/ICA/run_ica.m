% Implementation of an ICA-based face recognition system
%
% See also: CREATEDATABASE, EIGENFACECORE, RECOGNITION, FASTICA
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function [] = run_ica(verbose)

TrainDatabasePath = '../../train_images/';
TestDatabasePath = '../../test_images/';

% create training database
[TrainFiles, T] = CreateDatabase(TrainDatabasePath);

% compute the ICA...

% A is the centered matrix that has the mean vector subtracted from each column
[A, m] = remmean(T);

% now use the centered matrix to compute icasig
[icasig] = fastica(A')';
ProjectedImages = icasig' * A;

% test each image in the test set
TestFiles = dir(strcat(TestDatabasePath, '/*.pgm'));
num_correct = 0;

for i = 1 : size(TestFiles, 1)
    % perform recognition algorithm
    strtest = strcat(TestDatabasePath, '/', TestFiles(i).name);
    j = Recognition(strtest, m, icasig, ProjectedImages);

    % print results
    if verbose
        fprintf('test image: \"%s\"\n', TestFiles(i).name);
        fprintf('       ICA: \"%s/%s\"\n', TrainFiles(j).class, TrainFiles(j).name);
        fprintf('\n');
    end

    % determine whether the algorithm was correct
    % assumes that filename is formatted as '{class}_{index}.ppm'
    tokens_test = strsplit(TestFiles(i).name, '_');

    if strcmp(TrainFiles(j).class, tokens_test{1})
        num_correct = num_correct + 1;
    end
end

success_rate = num_correct / size(TestFiles, 1) * 100;

if verbose
    fprintf('%d / %d matched, %.2f%%\n', num_correct, size(TestFiles, 1), success_rate);
else
    fprintf('%.2f\n', success_rate);
end
