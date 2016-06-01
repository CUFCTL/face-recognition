% A sample script, which shows the usage of functions, included in
% PCA-based face recognition system (Eigenface method)
%
% See also: CREATEDATABASE, EIGENFACECORE, RECOGNITION

% Original version by Amir Hossein Omidvarnia, October 2007
%                     Email: aomidvar@ece.ut.ac.ir                  

clear all
clc
close all

format long e

% user selects database paths
% TrainDatabasePath = uigetdir('C:\Documents and Settings\smithmc\Desktop\', 'Select training database path' );
% TestDatabasePath = uigetdir('C:\Documents and Settings\smithmc\Desktop\', 'Select test database path');

% fixed paths
 TrainDatabasePath = 'F:\Documents\499\feret_images\Scotts_Cropped\new\train';
 TestDatabasePath = 'F:\Documents\499\feret_images\Scotts_Cropped\new\test';


TrainFiles = dir(TrainDatabasePath);

prompt = {'Enter test image name (without the .ppm extension):'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {'1'};

% read in test image
TestNum   = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestNum),'.ppm');
im = imread(TestImage);

% create training database
T = CreateDatabase(TrainDatabasePath);

[m, A, Eigenfaces] = EigenfaceCore(T);

% perform recognition algorithm
OutputName = Recognition(TestImage, m, A, Eigenfaces); 

SelectedImage = strcat(TrainDatabasePath,'\',TrainFiles(OutputName).name);
SelectedImage = imread(SelectedImage);

% display training and matched images
image(im)
title('Test Image');
figure,image(SelectedImage);
title('Equivalent Image');

str = strcat('Test image is : ', char(TestNum), '.ppm');
disp(str)

str = strcat('Matched image is : ', TrainFiles(OutputName).name);
disp(str)