function [m, A, Eigenfaces] = EigenfaceCore(T)

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the mean image 
m = mean(T,2); % Computing the average face image m = (1/P)*sum(Tj's)    (j = 1 : P)
Train_Number = size(T,2);
%%%%%%%%%%%%%%%%%%%%%%%% Calculating the deviation of each image from mean image
fprintf('calculating deviation of each image from mean image\n');
A = [];  
A = zeros(size(T,1),size(T,2));
for i = 1 : Train_Number
    temp = double(T(:,i)) - m; % Computing the difference image for each image in the training set Ai = Ti - m
    A(:,i) = temp; % Merging all centered images
end


[U,S,V] = svd(A, 0);
%produces the "economy size"
%decomposition. If X is m-by-n with m > n, then only the
%first n columns of U are computed and S is n-by-n.
%For m <= n, SVD(X,0) is equivalent to SVD(X).

Covariance_matrix =V*(S^2)*V';

Eigenfaces = A * Covariance_matrix; % A: centered image vectors