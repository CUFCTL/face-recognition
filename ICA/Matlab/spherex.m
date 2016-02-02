% SPHEREX - spheres the training vector x.
%    Requires x, P, to be predefined, and defines mx, c, wz.

fprintf('\nsubtracting mean\n');
mx=mean(x');
x=x-(ones(P,1)*mx)';
fprintf('calculating whitening filter\n');
c=cov(x');
wz=2*inv(sqrtm(c));
fprintf('whitening\n');
x=wz*x;
