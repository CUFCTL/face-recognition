% Align a set of face images (the training set T1, T2, ... , TM )
%
% Description: This function reshapes all 2D images of the training database
% into 1D column vectors. Then, it puts these 1D column vectors in a row to
% construct 2D matrix 'X'. Each column of 'X' is a training image, which has been reshaped into a 1D vector.
% Also, P is the total number of MxN training images and C is the number of
% classes.
%
%
% Argument:     path                   - Path of the training database
%
% Returns:      Files                  - Column vector of image entries
%               X                      - A 2D matrix, containing all 1D image vectors.
%                                        The length of 1D column vectors is MN and 'X' will be a MNxP 2D matrix.
%               Class_number           - Number of classes
%
% See also: STRCMP, STRCAT, RESHAPE
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function [Files, X, Class_number] = CreateDatabase(path)

%%%%%%%%%%%%%%%%%%%%%%%% File management
Files = dir(path);
Files = Files(3 : size(Files,1));

labels = {};

for i = 1 : size(Files, 1)
    % determine label
    tokens = strsplit(Files(i).name, '_');
    label_name = tokens{1};
    Files(i).label = label_name;

    % add label to list if unique
    I = find(strcmp(labels, label_name));

    if length(I) == 0
        labels = [labels; {label_name}];
    end
end

Class_number = length(labels);

%%%%%%%%%%%%%%%%%%%%%%%% Construction of 2D matrix from 1D image vectors
% grab the first image to get the height and width info for all images
str = strcat(path, '/', Files(1).name);
img = imread(str);
[irow, icol] = size(img);

% allocate width*height rows and number of images columns
X = zeros(irow*icol, size(Files,1));

for i = 1 : size(Files, 1)
    str = strcat(path, '/', Files(i).name);
    img = imread(str);

    [irow, icol] = size(img);

    % Reshaping 2D images into 1D image vectors
    X(:, i) = reshape(img',irow*icol,1);
end
