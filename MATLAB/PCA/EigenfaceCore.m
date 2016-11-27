% Use Principle Component Analysis (PCA) to determine the most
% discriminating features between images of faces.
%
% Argument:      X                      - A 2D matrix, containing all 1D image vectors.
%                                         Suppose all P images in the training database
%                                         have the same size of MxN. So the length of 1D
%                                         column vectors is M*N and 'X' will be a MNxP 2D matrix.
%                                         The data should already be mean-subtracted.
%
% Returns:       Eigenfaces             - (M*Nx(P-1)) Eigen vectors of the covariance matrix of the training database
%
% See also: EIG
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function Eigenfaces = EigenfaceCore(X)

%%%%%%%%%%%%%%%%%%%%%%%% Snapshot method of Eigenface methos
% We know from linear algebra theory that for a PxQ matrix, the maximum
% number of non-zero eigenvalues that the matrix can have is min(P-1,Q-1).
% Since the number of training images (P) is usually less than the number
% of pixels (M*N), the most non-zero eigenvalues that can be found are equal
% to P-1. So we can calculate eigenvalues of X'*X (a PxP matrix) instead of
% X*X' (a M*NxM*N matrix). It is clear that the dimensions of X*X' is much
% larger that X'*X. So the dimensionality will decrease.

fprintf('calculating surrogate matrix\n');
L = X'*X; % L is the surrogate of covariance matrix C=X*X'.

fprintf('getting eigenvalues from surrogate\n');
[V, D] = eig(L); % Diagonal elements of D are the eigenvalues for both L=X'*X and C=X*X'.

%%%%%%%%%%%%%%%%%%%%%%%% Sorting and eliminating eigenvalues
% All eigenvalues of matrix L are sorted and those who are less than a
% specified threshold, are eliminated. So the number of non-zero
% eigenvectors may be less than (P-1).

fprintf('eliminating eigenvalues <= 1\n');
L_eig_vec = V;% [];
%for i = 1 : size(V,2)
 %   if( D(i,i)>1 )
  %      L_eig_vec = [L_eig_vec V(:,i)];
   % end
%end

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the eigenvectors of covariance matrix 'C'
% Eigenvectors of covariance matrix C (or so-called "Eigenfaces")
% can be recovered from L's eiegnvectors.
fprintf('calculating eigenfaces\n');
Eigenfaces = X * L_eig_vec;
