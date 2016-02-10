% A modification of the sample script that loops through and matches
% all image files in the TestDatabasePath directory
%
% See also: CREATEDATABASE, EIGENFACECORE, RECOGNITION

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir                  

clear all
clc
close all

% You can customize and fix initial directory paths
% TrainDatabasePath = uigetdir('D:\Program Files\MATLAB\R2006a\work', 'Select training database path' );
% TestDatabasePath = uigetdir('D:\Program Files\MATLAB\R2006a\work', 'Select test database path');

% hard-coded directory can be replaced with pop-up above or changed here
 TrainDatabasePath = 'F:\Documents\499\feret_images\Scotts_Cropped\new4\train';
 TestDatabasePath = 'F:\Documents\499\feret_images\Scotts_Cropped\new4\test';

TrainFiles = dir(TrainDatabasePath);
TestFiles = dir(TestDatabasePath);

% create training database and build eigenface matrix
T = CreateDatabase(TrainDatabasePath);
[m, A, Eigenfaces] = EigenfaceCore(T);

for i = 1 : size(TestFiles,1)
    
    % remove semicolon to display all test file names as they're read in
    TestFiles(i).name;
    
    % ignore '.', '..', and 'Thumbs.db' (current directory "file", upper
    % director "file", and thumbnail database)
    if (strcmp(TestFiles(i).name,'.')||strcmp(TestFiles(i).name,'..')||strcmp(TestFiles(i).name,'Thumbs.db'))
        continue;
    end

    TestName = strcat('\',TestFiles(i).name);
    TestImage = strcat(TestDatabasePath,TestName);

    im = imread(TestImage);
    
    % perform recognition
    OutputName = Recognition(TestImage, m, A, Eigenfaces);
    
    SelectedImage = strcat(TrainDatabasePath,'\',TrainFiles(OutputName).name);
    SelectedImage = imread(SelectedImage);

    % output results (image numbers in file names should match)
    str = strcat('Test image is      : ',TestFiles(i).name);
    disp(str)

    str = strcat('Matched image is : ', TrainFiles(OutputName).name);
    disp(str)    
    
    str = ' ';
    disp(str)
end