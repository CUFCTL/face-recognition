%function C = loadFaceMat(imgdir); (Robust version)
%
%Loads a directory of images into the rows of C.
%imgdir is a string with the path to the directory containing the images.

imgdir = '../../orl_faces_ppm/';
TrainFiles = dir(strcat(imgdir, '/*.ppm'));

C = [];
for i = 1 : size(TrainFiles, 1)
    img = imread(strcat(imgdir, '/', TrainFiles(i).name));
    img = rgb2gray(double(img));
    img = reshape(img, 1, size(img,1) * size(img,2));
    C = [C; img];
end
