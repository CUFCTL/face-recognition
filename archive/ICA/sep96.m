% sep96.m implements the learning rule described in Bell \& Sejnowski, Vision
% Research, in press for 1997, that contained the natural gradient (w'w).
%
% Bell & Sejnowski hold the patent for this learning rule. 
%
% SEP goes once through the mixed signals, x 
% (which is of length M), in batch blocks of size B, adjusting weights,
% w, at the end of each block.
% sepout is called every F counts.
%
% I suggest a learning rate (lrate) of 0.006, and a blocksize (B) of 
% 300, at least for 2->2 separation.
% When annealing to the right solution for 10->10, however, lrate of
% less than 0.0001 and B of 10 were most successful.
%
% Copyright 1996 Tony Bell
% This may be copied for personal or academic use.
% For commercial use, please contact Tony Bell 
% (tony@salk.edu) for a commercial license.

x=x(:,perm);
sweep=sweep+1; t=1;
noblocks=fix(P/B);
BI=B*Id;
for t=t:B:t-1+noblocks*B,
  count=count+B;
  u=w*x(:,t:t+B-1); 
  w=w+L*(BI+(1-2*(1./(1+exp(-u))))*u')*w;
  if count>F, sepout; count=count-F; end;
end;
