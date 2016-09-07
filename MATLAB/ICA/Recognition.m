% Recognizing step....
%
% Description: This function compares two faces by projecting the images into facespace and
% measuring the Euclidean distance between them.
%
% Argument:      TestImage              - Path of the input test image
%
%                m                      - (M*Nx1) Mean of the training
%                                         database, which is output of 'EigenfaceCore' function.
%
%                Eigenfaces             - (M*Nx(P-1)) Eigen vectors of the
%                                         covariance matrix of the training
%                                         database, which is output of 'EigenfaceCore' function.
%
%                ProjectedImages        - ((P-1)xP) Matrix of projected image vectors.
%
% Returns:       OutputName             - Name of the recognized image in the training database.
%
% See also: RESHAPE, STRCAT
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function OutputName = Recognition(TestImage, m, Eigenfaces, ProjectedImages)

%%%%%%%%%%%%%%%%%%%%%%%% Projecting centered image vectors into facespace
% All centered images are projected into facespace by multiplying in
% Eigenface basis's. Projected vector of each face will be its corresponding
% feature vector.

Train_Number = size(Eigenfaces,2);
Test_Number = size(ProjectedImages,2);

%%%%%%%%%%%%%%%%%%%%%%%% Extracting the PCA features from test image
InputImage = imread(TestImage);
temp = InputImage(:,:,1);
[irow, icol] = size(temp);
InImage = reshape(temp',irow*icol,1);
Difference = double(InImage)-m; % Centered test image

%%% FPGA

ProjectedTestImage = Eigenfaces'*Difference; % Test image feature vector

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Euclidean distances
% Euclidean distances between the projected test image and the projection
% of all centered training images are calculated. Test image is
% supposed to have minimum distance with its corresponding image in the
% training database.

Euc_dist = zeros(Test_Number, 1);

for k = 1 : Test_Number
    Euc_dist(k) = norm(ProjectedTestImage - ProjectedImages(:, k)) ^ 2;
end

%%% FPGA

[Euc_dist_min , OutputName] = min(Euc_dist);
