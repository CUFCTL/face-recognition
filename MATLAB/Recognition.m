% This function compares two faces by projecting the images into facespace and
% measuring the Euclidean distance between them.
%
% Argument:      W                      - Projection matrix
%
%                P                      - Matrix of projected image vectors
%
% Returns:       rec_index              - Index of the recognized image in the training set.
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function rec_index = Recognition(P, p_test)

Test_Number = size(P, 2);

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Euclidean distances
% Euclidean distances between the projected test image and the projection
% of all centered training images are calculated. Test image is
% supposed to have minimum distance with its corresponding image in the
% training database.

Euc_dist = zeros(Test_Number, 1);

for i = 1 : Test_Number
    Euc_dist(i) = norm(p_test - P(:, i));
end

[~, rec_index] = min(Euc_dist);
