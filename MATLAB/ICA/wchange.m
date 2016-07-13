% Calculates stats on most recent weight change - magnitude and angle between
% old and new weight. 

function [change,delta,angle]=wchange(w,oldw,olddelta)
  [M,N]=size(w); delta=reshape(oldw-w,1,M*N);
  change=delta*delta';
  angle=acos((delta*olddelta')/sqrt((delta*delta')*(olddelta*olddelta')));
