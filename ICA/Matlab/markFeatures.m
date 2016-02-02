% Script markFeatures (Robust Version)
% Modified by William Halsey
% May 29, 2013
%
% This script is for marking eye and mouth positions in face images. It 
% writes a file called Labels.mat or TestLabels.mat, in which each row
% indexes an image, and the columns are [x,y] positions of subject's right 
% eye, [x,y] left eye, and [x,y] of mouth.
% 
% When prompted, the user must specify the directory that the images are
% in. This script is used for both the training and test images, and the
% user will be asked to specify which set at the end. The user will then be
% prompted to specify the location for the file (either Labels.mat or
% TestLabels.mat).
%
% For following code (in different files) ensure that the images used use
% the file extension .ppm


% The constants "START_ITER" and "END_ITER" are different for Macs and PC's
% because each architecture has a different number of hidden files in all
% directories including the image directories. If the preset numbers do not
% work try tweaking the values that correspond to your machine.
if ispc == 1
    START_ITER = 3;
    END_ITER = 0;

elseif ismac == 1
    START_ITER = 4;
    END_ITER = 1;

else
    fprintf('Error determining computer type! Check markFeatures.m\n');
end

imgdir = uigetdir();

cd (imgdir)
r = dir;

% get marks
marks = [];
for i = START_ITER:(size(r,1) - END_ITER)
    t = r(i).name;
    
    % The following line of code can be uncommented for debugging purposes.
    % Make sure that all printed filenames end with the extension .ppm.
%   fprintf('file name is %s\n', t);
    
    [X,map] = imread(t);

    figure(1);
    colormap gray;
    if isfloat(X)
        image(gray2ind(mat2gray((X))));
    else
        image(X);
    end
    title(t);
    disp 'Click subjects right eye, left eye, then mouth.'
    [m,n] = ginput(3); pos = round([m,n]);
    pos = reshape(pos',1,6);
    marks = [marks; pos];
end

destdir = uigetdir();
cd (destdir)

if menu('What is this for?', 'Test Images', 'Train Images') == 1
    save TestLabels marks r
    load TestLabels;
else
    save Labels marks r
    load Labels;
end