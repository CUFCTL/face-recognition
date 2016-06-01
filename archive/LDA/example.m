% A sample script, which shows the usage of functions, included in
% FLD-based face recognition system (Fisherface method)
%
% See also: CREATEDATABASE, FISHERFACECORE, RECOGNITION

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir   

% load_stuff: flag
%

%Good Reference? :-->http://www.eecs.umich.edu/~silvio/teaching/lectures/lecture22.pdf

clear all
clc
close all

% The TrainDatabase is what algorithm uses to learn?
TrainDatabasePath = fullfile('..','LDAIMAGES','Change','train');
% The TestDatabase is what images are being tested
TestDatabasePath = fullfile('..','LDAIMAGES','Test3');

load_stuff = 0; % need to run only the first time after you change the database

if (load_stuff == 0)
    %fprintf('Database, T:\n');
    T = CreateDatabase(TrainDatabasePath);
    [m V_PCA V_Fisher ProjectedImages_Fisher] = FisherfaceCore(T);
    save output_faces.mat T m V_PCA V_Fisher ProjectedImages_Fisher;
else
    load output_faces.mat *;
end

%return;

pass = 0;
fail = 0;
for i=1:120 % this is for using Test2 database
    TestImage = strcat(TestDatabasePath,'\',num2str(i),'.ppm');

    OutputNumber = Recognition(TestImage, m, V_PCA, V_Fisher, ProjectedImages_Fisher);

    strtest = strcat(TestDatabasePath, '\', num2str(i),'.ppm');
    im = imread(strtest);
    
     if(i==1)
         imshow(im);
     else
         figure,imshow(im);
     end
     title(strcat('Test Image',{' '}, num2str(i)));

    strtrain = strcat(TrainDatabasePath, '\', OutputNumber,'.ppm');

     im2 = imread(strtrain);
     figure,imshow(im2);
     title(strcat('Equivalent Image',{' '}, OutputName, '==', num2str(i)));

    % New subject every 4 images? Doesn't even really match
    if(i == (OutputNumber/4)+1)
        pass = pass + 1;
    else
        fail = fail + 1;
    end
    str = strcat('Test', num2str(i), {': '}, num2str(i), {'.ppm == '},' OutputNumber: ',int2str(OutputNumber),'.ppm', ' <--In Training Set');
    disp(str);
end

fprintf('%i passed, %i failed\n', pass, fail);