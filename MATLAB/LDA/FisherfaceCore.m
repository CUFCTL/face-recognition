% Use Principle Component Analysis (PCA) and Fisher Linear Discriminant (FLD) to determine the most
% discriminating features between images of faces.
%
% Description: This function gets a 2D matrix, containing all training image vectors
% and returns 4 outputs which are extracted from training database.
% Suppose Ti is a training image, which has been reshaped into a 1D vector.
% Also, P is the total number of MxN training images and C is the number of
% classes. At first, centered Ti is mapped onto a (P-C) linear subspace by V_PCA
% transfer matrix: Zi = V_PCA * (Ti - m_database).
% Then, Zi is converted to Yi by projecting onto a (C-1) linear subspace, so that
% images of the same class (or person) move closer together and images of difference
% classes move further apart: Yi = V_Fisher' * Zi = V_Fisher' * V_PCA' * (Ti - m_database)
%
% Argument:      X                      - (M*NxP) A 2D matrix, containing all 1D image vectors.
%                                         All of 1D column vectors have the same length of M*N
%                                         and 'X' will be a MNxP 2D matrix.
%                NumberClasses          - number of classes
%
% Returns:       W_lda                  - Projection matrix
%
% See also: EIG
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
function W_lda = FisherfaceCore(X, NumberClasses)

TotalImages = size(X, 2);
ClassSize = TotalImages / NumberClasses;   % Number of images per individual
NumEigenvaluesUsed = TotalImages - NumberClasses;
NumEigenvaluesFisher = NumberClasses - 1;

%%%%%%%%%%%%%%%%%%%%%%%% Snapshot method of Eigenface algorithm
% L is the surrogate of covariance matrix C = X * X'
L = X' * X;

% Diagonal elements of D are the eigenvalues for both L = X' * X and C = X * X'
[V, D] = eig(L);

% Flip left-right to place largest eigenvalues in the leftmost column
V = fliplr(V);
D = fliplr(D);

%%%%%%%%%%%%%%%%%%%%%%%% Sorting and eliminating small eigenvalues
L_eig_vec = V(:, 1 : NumEigenvaluesUsed);

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the eigenvectors of covariance matrix 'C'
V_PCA = X * L_eig_vec;

%%%%%%%%%%%%%%%%%%%%%%%% Projecting centered image vectors onto eigenspace
P_PCA = V_PCA' * X;

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the mean of each class in eigenspace
m_PCA = mean(P_PCA, 2); % Total mean in eigenspace
m = zeros(NumEigenvaluesUsed,NumberClasses);
Sw = zeros(NumEigenvaluesUsed,NumEigenvaluesUsed); % ;Initialization of Within Scatter Matrix
Sb = zeros(NumEigenvaluesUsed,NumEigenvaluesUsed); % Initialization of Between Scatter Matrix

for i = 1 : NumberClasses
    m(:,i) = mean( ( P_PCA(:,((i-1)*ClassSize+1):i*ClassSize) ), 2 )';

    S  = zeros(NumEigenvaluesUsed,NumEigenvaluesUsed);
    for j = ( (i-1)*ClassSize+1 ) : ( i*ClassSize )
        S = S + (P_PCA(:,j)-m(:,i))*(P_PCA(:,j)-m(:,i))';
    end

    Sw = Sw + S; % Within Scatter Matrix
    Sb = Sb + (m(:,i)-m_PCA) * (m(:,i)-m_PCA)'; % Between Scatter Matrix
end

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Fisher discriminant basis's
% We want to maximize the Between Scatter Matrix, while minimising the
% Within Scatter Matrix. Thus, a cost function J is defined, so that this condition is satisfied.
[J_eig_vec, J_eig_val] = eig(Sb,Sw); % Cost function J = inv(Sw) * Sb

J_eig_vec = fliplr(J_eig_vec);

%%%%%%%%%%%%%%%%%%%%%%%% Eliminating zero eigens and sorting in descend order
V_Fisher = J_eig_vec(:, 1 : NumEigenvaluesFisher);

% compute final projection matrix
W_lda = V_PCA * V_Fisher;
