%function Ctest = loadTestMat(imgdir); (Robust version)
%
%Loads a directory of images into the rows of C.
%imgdir is a string with the path to the directory containing the images.

imgdir = '../../orl_faces_ppm/';
TestFiles = dir(strcat(imgdir, '/*.ppm'));

Ctest = [];
for i = 1 : size(TestFiles, 1)
    img = imread(strcat(imgdir, '/', TestFiles(i).name));
    img = rgb2gray(double(img));
    img = reshape(img, 1, size(img,1) * size(img,2));
    Ctest = [Ctest; img];
end
