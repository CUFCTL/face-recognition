% For reports on training progress

[change,olddelta,angle]=wchange(oldw,w,olddelta); 
oldw=w;
fprintf('****sweep=%d, change=%.4f angle=%.1f deg., [N%d,M%d,P%d,B%d,L%.5f]\n', sweep,change,180*angle/pi,N,M,P,B,L);
