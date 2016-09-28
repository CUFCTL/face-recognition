% Align a set of face images (the training set T1, T2, ... , TM )
%
% Description: This function reshapes all 2D images of the training database
% into 1D column vectors. Then, it puts these 1D column vectors in a row to
% construct 2D matrix 'T'.
%
%
% Argument:     TrainDatabasePath      - Path of the training database
%
% Returns:      TrainFiles             - Column vector of image entries
%               T                      - A 2D matrix, containing all 1D image vectors.
%                                        Suppose all P images in the training database
%                                        have the same size of MxN. So the length of 1D
%                                        column vectors is MN and 'T' will be a MNxP 2D matrix.
%
% See also: STRCMP, STRCAT, RESHAPE
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function [TrainFiles, T] = CreateDatabase(TrainDatabasePath)

%%%%%%%%%%%%%%%%%%%%%%%% File management
ClassDirs = dir(TrainDatabasePath);
ClassDirs = ClassDirs(3 : size(ClassDirs,1));

TrainFiles = [];

for i = 1 : size(ClassDirs,1)
    str = strcat(TrainDatabasePath, ClassDirs(i).name, '/*.pgm');
    entries = dir(str);
    for j = 1 : size(entries,1)
        entries(j).class = ClassDirs(i).name;
    end

    TrainFiles = [TrainFiles; entries];
end

%%%%%%%%%%%%%%%%%%%%%%%% Construction of 2D matrix from 1D image vectors
% grab the first image to get the height and width info for all images
str = strcat(TrainDatabasePath, '/', TrainFiles(1).class, '/', TrainFiles(1).name);
img = imread(str);
[irow, icol] = size(img);

% allocate width*height rows and number of images columns
T = zeros(irow*icol, size(TrainFiles,1));

for i = 1 : size(TrainFiles,1)
    str = strcat(TrainDatabasePath, '/', TrainFiles(i).class, '/', TrainFiles(i).name);
    img = imread(str);

    [irow, icol] = size(img);

    % Reshaping 2D images into 1D image vectors
    T(:, i) = reshape(img',irow*icol,1);
end
