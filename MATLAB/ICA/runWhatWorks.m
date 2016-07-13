% runWhatWorks.m (Robust Version)
%
% written by: William Halsey
%
% Description: This code is to help facilitate the execution and
% interpretation of MATLAB code that implements the ICA algorithm on sets
% of images. This file is simply a script that will call the other MATLAB
% code ICA scripts in the order derived by myself and Scott Rodgers.
%

input = menu('Do you wish to run this script for the training data, test data, or both?', 'Training Data', 'Test Data', 'Both');

inputskip = menu('Do you want to align / mark faces?', 'Yes', 'No');
    input2 = menu('Do you wish to run markFeatures.m?', 'Yes', 'No');
if (input2 == 1 && inputskip ~= 1)
    if input == 3
        fprintf('Running markFeatures for Training Data...\n');
        markFeatures
        fprintf('Success! Moving on...\n\n');
        
        fprintf('Running markFeatures for Test Data...\n');
        markFeatures
        fprintf('Success! Moving on...\n\n');
    else
        fprintf('Running markFeatures...\n');
        markFeatures
        fprintf('Success! Moving on...\n\n');
    end
else
    fprintf('loading labels/TestLabels.mat...\n');
    load Labels
    load TestLabels
    fprintf('Success! Moving on...\n\n');
end

if (input == 1 && inputskip ~= 1)
    fprintf('Running align_Faces...\n');
    align_Faces
%   fprintf('\n\nno align_Faces yet\n\n');
    fprintf('Program ended successfully!\n\n');
    
elseif (input == 2 && inputskip ~= 1)
    fprintf('Running alignTestFaces...\n');
    alignTestFaces
 %  fprintf('\n\nno alignTestFaces yet\n\n');
    fprintf('Program ended successfully!\n\n');    
    
elseif (input == 3 && inputskip ~= 1)
    fprintf('Running align_Faces...\n');
    align_Faces
%   fprintf('\n\nno align_Faces yet\n\n');
    fprintf('Success! Moving on...\n\n');
    
    fprintf('Running alignTestFaces...\n');
    alignTestFaces
%   fprintf('\n\nno alignTestFaces yet\n\n');
    fprintf('Success! Moving on...\n\n');
  else
        fprintf('skipped align/mark faces\n');
end  
	
	
    input3 = menu('Do you want to run ICA I, ICA II, or Both', 'ICA I', 'ICA II', 'Both');
    if input3 == 1
        fprintf('Starting ICA I...\n\n');
        Arch1
%       fprintf('\n\nArch1 not ready yet\n\n');
        fprintf('\n\nICA I completed successfully!!\n\n');
        
    elseif input3 == 2
        fprintf('Starting ICA II...\n\n');
        Arch2
%       fprintf('\n\nArch2 not ready yet\n\n');
        fprintf('\n\nICA II completed successfully!!\n\n');
    elseif input3 == 3
        fprintf('Starting ICA I & II...\n\n');
%       Arch1and2
        %fprintf('\n\nArch1and2 not ready yet\n\n');
        fprintf('\n\nICA I & II completed successfully!!\n\n');
    else
        fprintf('Error with input\n');
    end
    
%else
 %   disp 'Error in the menu options... Try again.'
% end

