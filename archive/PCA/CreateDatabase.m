% Align a set of face images (the training set T1, T2, ... , TM )
%
% Description: This function reshapes all 2D images of the training database
% into 1D column vectors. Then, it puts these 1D column vectors in a row to
% construct 2D matrix 'T'.
%
%
% Argument:     TrainDatabasePath      - Path of the training database
%
% Returns:      T                      - A 2D matrix, containing all 1D image vectors.
%                                        Suppose all P images in the training database
%                                        have the same size of MxN. So the length of 1D
%                                        column vectors is MN and 'T' will be a MNxP 2D matrix.
%
% See also: STRCMP, STRCAT, RESHAPE

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function T = CreateDatabase(TrainDatabasePath)

%%%%%%%%%%%%%%%%%%%%%%%% File management
TrainFiles = dir(strcat(TrainDatabasePath, '/*.ppm'));
Train_Number = 0;
%for i = 1:size(TrainFiles,1)
%    if not(strcmp(TrainFiles(i).name,'.')|strcmp(TrainFiles(i).name,'..')|strcmp(TrainFiles(i).name,'Thumbs.db'))
%        Train_Number = Train_Number + 1; % Number of all images in the training database
%    end
%end

%%%%%%%%%%%%%%%%%%%%%%%% Construction of 2D matrix from 1D image vectors
%grab the first image to get the height and width info for all images
str = strcat('/',TrainFiles(1).name);
str = strcat(TrainDatabasePath,str);
img = imread(str);
[irow, icol] = size(img);
%allocate width*height rows and number of images columns
T = zeros(irow*icol/3, size(TrainFiles,1));

for j = 1 : 1
for i = 1 : size(TrainFiles,1)
    % I have chosen the name of each image in databases as a corresponding
    % number. However, it is not mandatory!
    TrainFiles(i).name;
    Train_Number = Train_Number + 1; % Number of all images in the training database
    str = strcat(TrainDatabasePath,'/',TrainFiles(i).name);
    img = imread(str);
    img = .2989*img(:,:,1) + .5870*img(:,:,2) + .1140*img(:,:,3);

    [irow, icol] = size(img);

    temp = reshape(img',irow*icol,1);   % Reshaping 2D images into 1D image vectors
    T(:,Train_Number) = temp; %add the image to the column and increment to the next image
end
Train_Number
end
