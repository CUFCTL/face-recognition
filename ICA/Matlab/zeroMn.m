%function Xzm = zeroMn(X)
%Returns a zero-mean form of the matrix X. Each row of Xzm will have 
%zero mean, same as in spherex.m. For PCA, put the observations in cols
%before doing zeroMn(X).

function Xzm = zeroMn(X)

[N,P] = size(X);
mx=mean(X');
Xzm=X-(ones(P,1)*mx)';

