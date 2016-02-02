%function Ctest = loadTestMat(imgdir); (Robust version)
%
%Loads a directory of images into the rows of C.
%imgdir is a string with the path to the directory containing the images.
%For example:

currentdir = pwd;

if ispc == 1
    imgdir = strcat(currentdir, '\AlignedTestImages\');
    
    START_ITER = 3;
    END_ITER = 0;

elseif ismac == 1
    imgdir = strcat(currentdir, '/AlignedTestImages/');
    
    START_ITER = 4;
    END_ITER = 1;

else
    fprintf('Error determining computer type! Check loadTestMat.m\n');
end

cd (imgdir)
r = dir;

Ctest = [];
for i = START_ITER:(size(r,1) - END_ITER)  %Wm--> changed initial value from 3 to 4 (may be mac specific)
   t = r(i).name;
        I=imread(t);
      tmp=mat2gray(double(I));
      tmp = reshape(tmp,1,size(tmp,1)*size(tmp,2));
      Ctest = [Ctest;tmp];      %Wm --> see next line for change
      %Ctest = vertcat(Ctest, tmp);  %Wm--> invalid dimensions for vertcat
      
      %Ctest = horzcat(Ctest, tmp);
end

cd (currentdir)
