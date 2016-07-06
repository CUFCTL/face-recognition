% A sample script, which shows the usage of functions, included in
% FLD-based face recognition system (Fisherface method)
%
% See also: CREATEDATABASE, FISHERFACECORE, RECOGNITION
%
% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir
%
% load_stuff: flag
%
%
% Good Reference? : http://www.eecs.umich.edu/~silvio/teaching/lectures/lecture22.pdf
%
clear
clc
close all

TrainDatabasePath = '../../orl_faces_ppm/';
TestDatabasePath = '../../orl_faces_ppm/';

load_stuff = 0; % need to run only the first time after you change the database

if (load_stuff == 0)
    T = CreateDatabase(TrainDatabasePath);
    [m, V_PCA, V_Fisher, ProjectedImages_Fisher] = FisherfaceCore(T);
    save output_faces.mat T m V_PCA V_Fisher ProjectedImages_Fisher;
else
    load output_faces.mat *;
end

TrainFiles = dir(strcat(TrainDatabasePath, '/*.ppm'));
TestFiles = dir(strcat(TestDatabasePath, '/*.ppm'));

pass = 0;
fail = 0;
for i = 1 : size(TestFiles, 1)
    strtest = strcat(TestDatabasePath, '/', TestFiles(i).name);

    OutputNumber = Recognition(strtest, m, V_PCA, V_Fisher, ProjectedImages_Fisher);

    im = imread(strtest);

%    if(i==1)
%        imshow(im);
%    else
%        figure,imshow(im);
%    end
%    title(strcat('Test Image',{' '}, num2str(i)));

    strtrain = strcat(TrainDatabasePath, '/', TrainFiles(OutputNumber).name);

%    im2 = imread(strtrain);
%    figure,imshow(im2);
%    title(strcat('Equivalent Image',{' '}, TrainFiles(OutputNumber).name, '==', num2str(i)));

    if ( i == OutputNumber )
        pass = pass + 1;
    else
        fail = fail + 1;
    end

    fprintf('Test: %s, OutputName: %s\n', strtest, strtrain);
end

fprintf('%i passed, %i failed\n', pass, fail);
