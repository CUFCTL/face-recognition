% Align a set of face images (the training set T1, T2, ... , TM )
%
% Description: This function reshapes all 2D images of the training database
% into 1D column vectors. Then, it puts these 1D column vectors in a row to
% construct 2D matrix 'T'. Each column of 'T' is a training image, which has been reshaped into a 1D vector.
% Also, P is the total number of MxN training images and C is the number of
% classes.
%
%
% Argument:     TrainDatabasePath      - Path of the training database
%
% Returns:      T                      - A 2D matrix, containing all 1D image vectors.
%                                        The length of 1D column vectors is MN and 'T' will be a MNxP 2D matrix.
%
% See also: STRCMP, STRCAT, RESHAPE
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function T = CreateDatabase(TrainDatabasePath)

%%%%%%%%%%%%%%%%%%%%%%%% File management

TrainFiles = dir(strcat(TrainDatabasePath, '/*.ppm'));
Train_Number = size(TrainFiles, 1);

%%%%%%%%%%%%%%%%%%%%%%%% Construction of 2D matrix from 1D image vectors
T = [];
for i = 1 : Train_Number
    str = strcat(TrainDatabasePath, '/', TrainFiles(i).name);

    img = imread(str);
    img = rgb2gray(img);

    [irow, icol] = size(img);

    temp = reshape(img',irow*icol,1);   % Reshaping 2D images into 1D image vectors
    T = [T temp]; % 'T' grows after each turn
end

T = double(T);
