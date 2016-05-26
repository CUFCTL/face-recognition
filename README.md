# Clemson FCT Facial Recognition

## The Matrix Library

Function Name              | PCA | LDA | ICA | Verification Status | Verify Date | Member
---                        |:---:|:---:|:---:|---                  |---          |---       
m_initialize               |  x  |  x  |  x  | Verified            | 10/21/15    | Taylor
m_copy                     |     |     |  x  | Verified            | 10/21/15    | Taylor
m_free                     |  x  |  x  |  x  | Verified            | 10/21/15    | Taylor
m_fprint                   |     |  x  |     | Verified            | 10/21/15    | Taylor
m_fwrite                   |  x  |     |     | Verified            | 10/21/15    | Taylor
m_fscan                    |     |     |     | Verified            | 10/21/15    | Taylor
m_fread                    |  x  |     |     | Verified            | 10/21/15    | Taylor
                           |     |     |     |                     |             |
m_flipCols                 |     |     |  x  | Verified            | 10/02/15    | James
m_normalize                |     |     |     | Verified            | 10/02/2015  | James
m_elem_mult                |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_truncate            |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_divideByConst       |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_acos                |     |     |     | Verified            | 10/02/2015  | James
m_elem_sqrt                |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_negate              |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_exp                 |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_pow                 |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_divideByMatrix      |     |     |  x  | Verified            | 10/02/2015  | James
m_elem_add                 |     |     |  x  | Verified            | 10/02/2015  | James
m_sumCols                  |     |     |  x  | Verified            | 10/02/2015  | James
m_meanCols                 |     |  x  |     | Verified            | 10/02/2015  | James
m_sumRows                  |     |     |     | Verified            | 10/02/2015  | James
m_meanRows                 |  x  |     |     | Verified            | 10/06/2015  | James
m_findNonZeros             |     |     |     | Verified            | 10/06/2015  | James
m_normalize_columns        |  x  |     |     | Verified            |             |
m_transpose                |  x  |  x  |  x  | Verified            | 10/06/2015  | James
m_reshape                  |     |     |     | Verified            | 10/06/2015  | James
                           |     |     |     |                     |             |
m_inverseMatrix            |     |     |  x  | Not Verified        | 10/07/15    | Miller
m_norm                     |     |     |     | Not Verified        | 10/07/15    | Miller
m_sqrtm                    |     |     |  x  | Not Verified        | 10/07/15    | Miller
m_determinant              |     |     |     | Not Verified        | 10/07/15    | Miller
m_cofactor                 |     |     |     | Not Verified        | 10/07/15    | Miller
m_covariance               |     |     |  x  | Verified            | 11/05/15    | Greg
                           |     |     |     |                     |             |
m_dot_subtract             |     |     |  x  | Verified            | 10/21/15    | Taylor
m_dot_add                  |     |     |  x  | Verified            | 10/21/15    | Taylor
m_dot_division             |     |     |  x  | Verified            | 11/03/15    | Greg
m_matrix_multiply          |  x  |  x  |  x  | Verified            | 10/21/15    | Taylor
m_matrix_division          |     |     |     | Not Verified        | 10/21/15    | Taylor
m_reorder_columns          |     |     |  x  | Not Verified        | 10/21/15    | Taylor
                           |     |     |     |                     |             |
m_eigenvalues_eigenvectors |  x  |  x  |  x  | Not Verified        | 10/22/15    | Colin
loadPPMtoMatrixCol         |  x  |     |     | Not Verified        | 10/22/15    | Colin
writePPMgrayscale          |  x  |     |     | Not Verified        | 10/22/15    | Colin

_Note: When we say verified we are talking about an initial verification step which does not mean the function is fully trustworthy in the final code. There will be an additional possibility for this column: "Verified using BLAS" which will mean that the library function has be implemented in blas and is ready to be used in the final code_

#### BLAS and LAPACK

Much of the code in this project depends on BLAS and LAPACK. In order to run it properly, it is necessary to install the following libraries:

    libblas-dev (1.2.20110419-5)
    libblas3 (1.2.20110419-5)
    libgfortran3 (4.8.1-10ubuntu9)
    liblapack-dev (3.4.2+dfsg-2)
    liblapack3 (3.4.2+dfsg-2)
    liblapacke (3.4.2+dfsg-2)
    liblapacke-dev (3.4.2+dfsg-2)

##### Ubuntu

    sudo apt-get install liblapacke-dev

##### Mac

Confirmed to run on Mac OS 10.11.1

Download LAPACK 3.5.0 at http://www.netlib.org/lapack/
Download BLAS 3.5.0 at http://www.netlib.org/blas/

Download gfortran 5.2 for OS X (10.10 - 10.11.x)
http://coudert.name/software/gfortran-5.2-Yosemite.dmg

Download gfortran 4.8.2 for OS X (10.7 - 10.9)
http://coudert.name/software/gfortran-4.8.2-MountainLion.dmg

###### BLAS

run 'make' inside the BLAS-3.5.0 directory

    mv blas-LINUX.a libblas.a
    sudo cp libblas.a /usr/local/lib/

###### LAPACK

open make.inc.example in a text editor
set BLASLIB on line 68 equal to ‘/usr/local/lib/libblas.a’

rename make.inc.example to make.inc

run 'make' inside the lapack-3.5.0 directory

    sudo cp liblapack.a /usr/local/lib

## The Algorithms

### ICA

TODO

### LDA

TODO

### PCA

To convert JPEG images to PPM with `ffmpeg`:

    cd [images-folder]
    for f in *.jpg
    do ffmpeg -i $f -s 300x200 "$(basename $f .jpg)".ppm
    done

To run PCA on a training set of PPM images:

    cd PCA
    make
    ./pca-train [training-images-folder]

To test a set of PPM images against the training set:

    ./pca-recognize [test-images-folder]
