/****************************************
Recognizing step....

Description: This function compares two faces by projecting the images into
facespace and measuring the Euclidean distance between them.

Argument:   TestImage              - Path of the input test image

            m_database             - (M*Nx1) Mean of the training database
                                     database, which is output of 'EigenfaceCore'
                                     function.

            V_PCA                  - (M*Nx(P-1)) Eigen vectors of the covariance
                                     matrix of the training database

            V_Fisher               - ((P-1)x(C-1)) Largest (C-1) eigen vectors
                                     of matrix J = inv(Sw) * Sb

            ProjectedImages_Fisher - ((C-1)xP) Training images, which
                                     are projected onto Fisher linear space

Returns:    OutputName             - Name of the recognized image in the
                                     training database.

See also: RESHAPE, STRCAT

Original version by Amir Hossein Omidvarnia, October 2007
                 Email: aomidvar@ece.ut.ac.ir
****************************************/

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <math.h>
#include "ppm.h"

// CHANGE ME
#define DATA_PATH "C:/Users/Guru/Documents/MATLAB/FCT Image Recognition/Saved Data Structures/m.mat"
#define DATA_PATH2 "C:/Users/Guru/Documents/MATLAB/FCT Image Recognition/Saved Data Structures/Inverse_V_Fisher.mat"
#define DATA_PATH3 "C:/Users/Guru/Documents/MATLAB/FCT Image Recognition/Saved Data Structures/Inverse_V_PCA.mat"
#define DATA_PATH4 "C:/Users/Guru/Documents/MATLAB/FCT Image Recognition/Saved Data Structures/ProjectedImages_Fisher.mat"
#define ROOT_DIRECTORY "C:/Users/Guru/Documents/MATLAB/FCT Image Recognition/Saved Data Structures/"
//#define DATA_PATH "C:/Users/Guru/Documents/MATLAB/FCT Image Recognition/T.mat"

using namespace std;

typedef struct {
    double **data;
    int rows, cols;
} MATRIX;

MATRIX m_database; //ends up being a 1d matrix
MATRIX Inverse_V_Fisher;
MATRIX Inverse_V_PCA;
MATRIX ProjectedImages_Fisher;
MATRIX Difference;
MATRIX v_fisherT_x_v_pcaT;
MATRIX ProjectedTestImage;

int MatrixRead_Binary(); // This version reads using fread() assumption:
        // matrix was written binary

int main()
{
    char filename[255] = {}; // "3.ppm"; // filename of the test image


    // We need to load the data structures that were created by the
            //CreateDatabase step
    if (!MatrixRead_Binary()) //Load structures
    {
        cerr << "Error!!!" << endl;
        exit(-1);
    }
    cout << "Database Files Loaded From Disk..." << endl << endl;
    ;
    PPMImage *TestImage; // pointer to the structure of our loaded test image

    int pass = 0, fail = 0; /*Let's keep a few stats*/

    for (int iterations = 1; iterations <= 30; iterations++) //994 is the number of test images that we have sequentially numbered
    {
        sprintf(filename, "..\\..\\..\\LDA Sept 08, 2009\\Test2\\%d.ppm", iterations); //concat our filename together.
        TestImage = ppm_image_constructor(filename);
        //grayscale(TestImage);
        cout << "Test Image Loaded From Disk and converted to grayscale..." << endl;
        /*Subtract mean of database from the test image*/

        //First lets allocate our difference array
        Difference.rows = m_database.rows;
        Difference.cols = 1;
        Difference.data = new double*[Difference.rows];
        for (int i = 0; i < Difference.rows; i++) {
            Difference.data[i] = new double[Difference.cols];
        }

        for (int i = 0; i < m_database.rows; i++) {
            Difference.data[i][0] = TestImage->pixels[i].r - m_database.data[i][0]; //mean database is a 1d vector    (using red channel for gray)
        }
        //Now lets multiply in the last matrix to calculate our ProjectedTestImage
        //v_fisherT_x_v_pcaT * Difference

        ProjectedTestImage.rows = v_fisherT_x_v_pcaT.rows;
        ProjectedTestImage.cols = Difference.cols;
        ProjectedTestImage.data = new double *[ProjectedTestImage.rows];
        for (int i = 0; i < ProjectedTestImage.rows; i++) //allocate memory for our new array
        {
            ProjectedTestImage.data[i] = new double[ProjectedTestImage.cols];
            if (ProjectedTestImage.data[i] == 0) {
                cout << "Dynamic Allocation Failed!!!" << endl;
                return -1;
            }
        }

        for (int i = 0; i < v_fisherT_x_v_pcaT.rows; i++) //perform matrix multiplication computation
        {
            for (int j = 0; j < Difference.cols; j++) //This loop executes once
            {
                ProjectedTestImage.data[i][j] = 0.0;
                for (int k = 0; k < v_fisherT_x_v_pcaT.cols; k++) {
                    ProjectedTestImage.data[i][j] += v_fisherT_x_v_pcaT.data[i][k] * Difference.data[k][j];
                }
            }
        }

        unsigned long int Train_Number = 0;
        Train_Number = ProjectedImages_Fisher.cols; //Satisfys line 27
        double *q = 0; //Holds a column vector
        q = (double *) malloc(ProjectedImages_Fisher.rows * sizeof (double));

        double * Euc_dist = (double *) malloc(sizeof (double) *ProjectedImages_Fisher.cols);
        double temp = 0;

        for (int i = 0; i < Train_Number; i++) //line 44 Recognition.m
        {
            for (int j = 0; j < ProjectedImages_Fisher.rows; j++) //create q
            {
                q[j] = ProjectedImages_Fisher.data[j][i];
            } //q has been populated

            /*At this point, ProjectedTestImage is 99x1 and q is 99x1  (Based on testing database)*/
            for (int x = 0; x < ProjectedImages_Fisher.rows; x++) {
                temp += ((ProjectedTestImage.data[x][0] - q[x])*(ProjectedTestImage.data[x][0] - q[x])); //line 46
            }
            Euc_dist[i] = temp;
            temp = 0; //reset our running count
        }
        //at this point, Euc_dist should be populated

        //we need to find the min euc_dist and its index
        //int Euc_dist_len=sizeof(*Euc_dist)/sizeof(double);
        int Euc_dist_len = ProjectedImages_Fisher.cols; //the length of euc_dist
        double min = Euc_dist[0];

        int Recognized_index = 0;
        for (int i = 0; i < Euc_dist_len; i++) {
            if (Euc_dist[i] < min) //reassign min
            {
                min = Euc_dist[i];
                Recognized_index = i;
            }
        }
        int filename_index = Recognized_index + 1; //because our files are named starting at 1 and not 0. Arrays start at 0 in c

        cout << "Test" << iterations << ": " << iterations << ".ppm == " << filename_index << ".ppm" << endl;
        cout << "-----------------------------------" << endl;

        //////////////for statistic tracking
        if (iterations == ((Recognized_index - 1) / 4 + 1))
            pass++;
        else
            fail++;
        //////////////
        ppm_image_destructor(TestImage, 0); //Free the loaded image
        //Need to free the memory that was allocated...

        //free Difference matrix
        for (int i = 0; i < Difference.rows; i++)
            delete[] Difference.data[i];
        delete[] Difference.data;

        //Free ProjectedTestImage matrix...
        for (int i = 0; i < ProjectedTestImage.rows; i++)
            delete[] ProjectedTestImage.data[i];
        delete[] ProjectedTestImage.data;

        //Free q vector
        free(q); //wha? why exactly did i use malloc?

        //Free Euc_dist
        free(Euc_dist); //huh?
        ///////////Allocated Memory Freed//////////////////////
    }
    cout << pass << " Correct " << fail << " Wrong" << endl;

    return 0;
}

/*MatrixRead_Binary() returns 0 on error and 1 on success*/
int MatrixRead_Binary() //Reads in all required matrices
{
    int rows, cols;
    /**********************Read In The m.mat*************************/
    ifstream fin;
    fin.open("m.mat", ios::in | ios::binary);
    if (!fin.is_open()) {
        cerr << "Unable to Open m.mat!!!" << endl;
        return 0;
    }
    fin.read((char*) &rows, sizeof (rows));
    fin.read((char*) &cols, sizeof (cols));
    m_database.data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        m_database.data[i] = new double[cols];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin.read((char*) &m_database.data[i][j], sizeof (m_database.data[i][j]));
        }
    }
    m_database.rows = rows;
    m_database.cols = cols;
    fin.close();
    /**************************************************************/

    /**************Read In The Inverse_V_Fisher.mat****************/
    fin.open("Inverse_V_Fisher.mat", ios::in | ios::binary);
    if (!fin.is_open()) {
        cerr << "Unable to Open Inverse_V_Fisher.mat!!!" << endl;
        return 0;
    }
    fin.read((char*) &rows, sizeof (rows));
    fin.read((char*) &cols, sizeof (cols));
    Inverse_V_Fisher.data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        Inverse_V_Fisher.data[i] = new double[cols];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin.read((char*) &Inverse_V_Fisher.data[i][j], sizeof (Inverse_V_Fisher.data[i][j]));
        }
    }
    Inverse_V_Fisher.rows = rows;
    Inverse_V_Fisher.cols = cols;
    fin.close();

    /**************************************************************/


    /**************Read In The Inverse_V_PCA.mat****************/
    fin.open("Inverse_V_PCA.mat", ios::in | ios::binary);
    if (!fin.is_open()) {
        cerr << "Unable to Open Inverse_V_PCA!!!" << endl;
        return 0;
    }
    fin.read((char*) &rows, sizeof (rows));
    fin.read((char*) &cols, sizeof (cols));
    Inverse_V_PCA.data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        Inverse_V_PCA.data[i] = new double[cols];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin.read((char*) &Inverse_V_PCA.data[i][j], sizeof (Inverse_V_PCA.data[i][j]));
        }
    }
    Inverse_V_PCA.rows = rows;
    Inverse_V_PCA.cols = cols;
    fin.close();

    /**************************************************************/

    /**************Read In The ProjectedImages_Fisher.mat****************/
    fin.open("ProjectedImages_Fisher.mat", ios::in | ios::binary);
    if (!fin.is_open()) {
        cerr << "Unable to Open ProjectedImages_Fisher.mat!!!" << endl;
        return 0;
    }
    fin.read((char*) &rows, sizeof (rows));
    fin.read((char*) &cols, sizeof (cols));
    ProjectedImages_Fisher.data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        ProjectedImages_Fisher.data[i] = new double[cols];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin.read((char*) &ProjectedImages_Fisher.data[i][j], sizeof (double));
        }
    }
    ProjectedImages_Fisher.rows = rows;
    ProjectedImages_Fisher.cols = cols;
    fin.close();

    /**************************************************************/

    /**************Read In The v_fisherT_x_v_pcaT.mat****************/
    fin.open("v_fisherT_x_v_pcaT.mat", ios::in | ios::binary);
    if (!fin.is_open()) {
        cerr << "Unable to Open v_fisherT_x_v_pcaT.mat!!!" << endl;
        return 0;
    }
    fin.read((char*) &rows, sizeof (rows));
    fin.read((char*) &cols, sizeof (cols));
    v_fisherT_x_v_pcaT.data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        v_fisherT_x_v_pcaT.data[i] = new double[cols];
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin.read((char*) &v_fisherT_x_v_pcaT.data[i][j], sizeof (v_fisherT_x_v_pcaT.data[i][j]));
        }
    }
    v_fisherT_x_v_pcaT.rows = rows;
    v_fisherT_x_v_pcaT.cols = cols;
    fin.close();
    /**************************************************************/

    //Now make sure everything is opened properly
    if (m_database.data == 0 || Inverse_V_Fisher.data == 0 || Inverse_V_PCA.data == 0 || ProjectedImages_Fisher.data == 0 || v_fisherT_x_v_pcaT.data == 0) {
        return 0; //Memory not allocated properly somewhere
    } else
        return 1;
}
