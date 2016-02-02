%function C = loadFaceMat(imgdir); (Robust version)
%
%Loads a directory of images into the rows of C.
%imgdir is a string with the path to the directory containing the images.
%For example:

currentdir = pwd;

if ispc == 1
    imgdir = strcat(currentdir, '\AlignedFaceImages\');
    
    START_ITER = 3;
    END_ITER = 0;

elseif ismac == 1
    imgdir = strcat(currentdir, '/AlignedFaceImages/');
    
    START_ITER = 4;
    END_ITER = 1;

else
    fprintf('Error determining computer type! Check loadTestMat.m\n');
end

cd (imgdir)
r = dir;

C = [];
for i = START_ITER:(size(r,1) - END_ITER)  %Wm--> change the initial value to 4 (may be mac specific problem)
   t = r(i).name;
        I=imread(t);
      tmp=mat2gray(double(I));
      tmp = reshape(tmp,1,size(tmp,1)*size(tmp,2));
      C = [C;tmp]; %Wm--> see loadTestMat.m for comments about this and
      %following line
      
      %C = horzcat(C, tmp);
end

cd (currentdir)
