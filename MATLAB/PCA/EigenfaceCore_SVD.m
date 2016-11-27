function Eigenfaces = EigenfaceCore_SVD(X)

[U,S,V] = svd(X, 0);
%produces the "economy size"
%decomposition. If X is m-by-n with m > n, then only the
%first n columns of U are computed and S is n-by-n.
%For m <= n, SVD(X,0) is equivalent to SVD(X).

Covariance_matrix =V*(S^2)*V';

Eigenfaces = X * Covariance_matrix;
