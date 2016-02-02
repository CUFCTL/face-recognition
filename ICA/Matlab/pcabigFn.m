%function [U,R,E] = pcabigFn(B);
%Compute PCA by calculating smaller covariance matrix and reconstructing
%eigenvectors of large cov matrix by linear combinations of original data
%given by the eigenvecs of the smaller cov matrix. 
%Data in Cols of B. Third version.  
%
%***** justification
%
%B = N x P data matrix.  N = dim of data  (Data in Cols of B, zero mean)
%                        P = #examples
%                        N >> P
%
%Want eigenvectors ui of BB' (NxN)
%Solution:
%Find eigenvectors vi of B'B (PxP)
%From def of eigenvector: B'Bvi = di vi ---> BB'Bvi = di Bvi
%Eigenvecs of BB' are Bvi
%-------------------------------
%[V,D] = eig (B'B)
%Eigenvecs are in cols of V.    (Sorted cols)
%
%U = BV;  Cols of U are Bvi (eigenvecs of lg cov mat.) (Gave unit length)
%R = B'U; Rows of R are pcarep of each observation.
%E = eigenvalues        (eigenvals of small and large cov mats are equal)
%*****

function [U,R,E] = pcabigFn(B)

%Read data into columns of B;
%B = datamat';
[N,P] = size(B);

%********subtract mean
mb=mean(B');
B=B-(ones(P,1)*mb)';

%********Find eigenvectors vi of B'B (PxP)
[V,D] = eig (1/(P-1)*(B'*B));   %scale factor gives eigvals correct
                                %magnitude for large cov mat 
                                %(assuming sample cov)
                                %(assuming sample cov)
%********Sort eigenvectors
eigvalvec = max(D);
[seigvals, index] = sort(eigvalvec); % sort goes low to high
Vsort = V(:,fliplr(index));

%********Reconstruct
U = B*Vsort;  % Cols of U are Bvi. (N-dim Eigvecs)

%********Give eigvecs unit length.  Improves classification.
length = sqrt (sum (U.^2));
U = U ./ (ones(N,1) * length);

R = B'*U;  % Rows of R are pcarep of each image.
E = fliplr(seigvals);
