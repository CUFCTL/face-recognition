% Script align_Faces (Robust version)
%
% Aligns the eye positions of a directory of face images. Reads in Labels.mat, 
% obtained using getLabels.m, writes jpegs to a specified directory.
% Specify directory paths at the top of the file. 
% LabelDir is where Labels.mat is.
% imgDir is where the original images are
% DestDir is where you want the cropped jpegs to go. 

if ispc == 1
    LabelDir = uigetdir();
    imgdir = strcat(LabelDir, '\test\');
    DestDir = strcat(LabelDir, '\AlignedTestImages\');
    
    START_ITER = 3;
    END_ITER = 0;
    MARK_OFF = 2;

elseif ismac == 1
    LabelDir = uigetdir();
    imgdir = strcat(LabelDir, '/test/');
    DestDir = strcat(LabelDir, '/AlignedTestImages/');
    
    START_ITER = 4;
    END_ITER = 1;
    MARK_OFF = 3;

else
    fprintf('Error determining computer type! Check alignTestFaces.m\n');
end
  
homeDir = pwd; 

% CHANGE THESE VARIABLES AS NEEDED
%XSIZE =  YSIZE = 	%Size of desired cropped image
%EYES = %Number of pixels desired between the eyes
%TEETH_EYES = %Desired no. of pixels from teeth to eyes. 

XSIZE = 240; YSIZE = 292;      
EYES = 130;
TEETH_EYES = 165; 
%SEE ALSO PARAMETERS IN CROP ROUTINE. EYES BELOW MIDPOINT.
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Here is where we load the images and do the preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      cd (LabelDir)
      load TestLabels
      cd (imgdir)
      r = dir;

for i = START_ITER:(size(r,1) - END_ITER)
   imgName = r(i).name;
   %[X,map] = imread([ t ]);
      I=imread(imgName);
      I=rgb2gray(I);

     %Extract face measurements:
     %dxeyes =  x distance between eyes in original image                     
             % Average the locations of inner & outer corners.
     %dyeyes =  y distance between eyes in original image 
     %dEeyes = Euclidean norm distance between eyes
     %dEteeth_eyes = Euclidean norm distance from teeth to midpt b/neyes
                 
     [height, width] = size(I);
     dxeyes = marks(i-MARK_OFF,3) - marks(i-MARK_OFF,1); %Check these. 
     dyeyes = marks (i-MARK_OFF,4) - marks(i-MARK_OFF,2);  %Wm --> changed lines 48, 49, 51, 52, and 53 to read i-3... instead of i-2...
     dEeyes = sqrt(dxeyes^2 + dyeyes^2);
     mean_eye_x = mean([marks(i-MARK_OFF,1), marks(i-MARK_OFF,3)]);
     mean_eye_y = mean([marks(i-MARK_OFF,2), marks(i-MARK_OFF,4)]);
     dEteeth_eyes = sqrt((marks(i-MARK_OFF,5)-mean_eye_x)^2 + (marks(i-MARK_OFF,6)-mean_eye_y)^2); 
                                          
     %scale
     yscale = TEETH_EYES / dEteeth_eyes; xscale = EYES / dEeyes;
     height_new = yscale*height; width_new = xscale*width;
     tmp0=imresize(I,[height_new,width_new],'bicubic');
     %figure(2);imshow(tmp0)

     %fprintf('yscale = %d, xscale = %d\n', yscale, xscale);
   
      %rotate (Problem: imrotate rotates about the center of the image. 
      %To avoid losing feature position information, must first center the 
      %image on the right eye before rotating. 
      %Then use right eye position to determine cropping.
      Reye_x = marks(i-MARK_OFF,1);        %Wm --> see comment in line 49 for canges in 64 and 65
      Reye_y = marks(i-MARK_OFF,2);

      %crop a 200x200 window centered on left eye:
      %Zero-pad to make sure window never falls outside of image. 
      %W = 100; %Window radius
      W = 500;  %For bigger images (Gwen's params).
      padcols = zeros(size(tmp0,1),W); padrows = zeros(W,size(tmp0,2)+2*W);
      padcols = uint8(padcols); padrows=uint8(padrows);
      tmp = [padrows;padcols,tmp0,padcols];

      tmpx = xscale*Reye_x - W +W; tmpy = yscale*Reye_y - W +W;
      %fprintf('tmpx = %d, xscale*reye_x = %d, tmpy = %d, yscale*reye_y = %d\n', tmpx, xscale*Reye_x, tmpy, yscale*Reye_y);
      tmp1 = imcrop(tmp,[tmpx,tmpy,2*W,2*W]);
      %figure(2);imshow(tmp1)

      angle = 180/pi*atan((yscale*dyeyes)/(xscale*dxeyes));
      tmp2 = imrotate(tmp1,angle,'bicubic','crop');
      %figure(2); imshow(tmp2);

      %crop
      % x and y give the upper left corner of cropped image
      % Reye is centered at (W,W) = (100,100).
      % For bigger images (W,W) = (500,500)
      x = W - (XSIZE-EYES)/2;
      %y = W - YSIZE/2;   %Eyes at midpoint
      y = W - YSIZE*1/3;  %Face box
      tmp3=imcrop(tmp2,[x,y,XSIZE,YSIZE]);
      figure(1); imshow(tmp3);

      fprintf('i = %d %d, tmp = %d %d, tmp0 = %d %d, tmp1 = %d %d, tmp2 = %d %d, tmp3 = %d %d\n', size(I), size(tmp), size(tmp0), size(tmp1), size(tmp2), size(tmp3))
      
      %save
      [imgName, R] = strtok(imgName, '.');
      fname = [DestDir,imgName, '.pgm'];
      imwrite(tmp3,fname,'pgm') 
end

cd (homeDir)
