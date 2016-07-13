% This script outputs the intermedate databases and objects used for
% recognition to a file that can be read by a c++ program


clear all
clc
close all

% You can customize and fix initial directory paths
TrainDatabasePath = 'C:\Users\Guru\Documents\Matlab\FCT Image Recognition\LDA Sept 08, 2009\Train2';
TestDatabasePath = 'C:\Users\Guru\Documents\Matlab\FCT Image Recognition\LDA Sept 08, 2009\Train2';

load_stuff = 0;

if(load_stuff == 0)
   
   T = CreateDatabase(TrainDatabasePath);
   [m V_PCA V_Fisher ProjectedImages_Fisher] = FisherfaceCore(T);
   Inverse_V_Fisher=V_Fisher';
   Inverse_V_PCA=V_PCA';
   
   %note that we do not need to write T to disk but we will anyway
 fprintf('Writing Data Objects To Disk\n')
      
   [rows cols]=size(T);
   handle=fopen('T.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%d ',T(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nT.mat Sucessfully Written\n')
   
   [rows cols]=size(m);
   handle=fopen('m.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%f ',m(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nm.mat Sucessfully Written\n')
   
   [rows cols]=size(V_PCA);
   handle=fopen('V_PCA.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%f ',V_PCA(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nV_PCA.mat Sucessfully Written\n')
   
   [rows cols]=size(V_Fisher);
   handle=fopen('V_Fisher.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%f ',V_Fisher(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nV_Fisher.mat Sucessfully Written\n')
   
   [rows cols]=size(ProjectedImages_Fisher);
   handle=fopen('ProjectedImages_Fisher.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%.19f ',ProjectedImages_Fisher(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nProjectedImages_Fisher.mat Sucessfully Written\n')
   
   %save output_faces.mat T m V_PCA V_Fisher ProjectedImages_Fisher
   
   [rows cols]=size(Inverse_V_Fisher);
   handle=fopen('Inverse_V_Fisher.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%f ',Inverse_V_Fisher(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nInverse_V_Fisher.mat Sucessfully Written\n')
   
   [rows cols]=size(Inverse_V_PCA);
   handle=fopen('Inverse_V_PCA.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%.19f ',Inverse_V_PCA(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nInverse_V_PCA.mat Sucessfully Written\n')
   
   
   v_fisherT_x_v_pcaT=V_Fisher' * V_PCA';  %this is an intermediate that will be used in the c code
   [rows cols]=size(v_fisherT_x_v_pcaT);
   handle=fopen('v_fisherT_x_v_pcaT.mat','w');
   %write the matrix size out first%
   fprintf(handle,'%d %d\n',rows,cols);
   for (i=1:rows)
       for (j=1:cols)
           fprintf(handle,'%d ',v_fisherT_x_v_pcaT(i,j));
       end
       fseek(handle,-1,'cof'); %move back 1 pos since there is a blank space there
       fprintf(handle,'\n'); %replace that blank space with a newline
   end
   fclose(handle);
   fprintf('\nv_fisherT_x_v_pcaT.mat Sucessfully Written\n')
   
   
   
   
else
   %load output_faces.mat *;
   fprintf('Not Implemented\n')
end