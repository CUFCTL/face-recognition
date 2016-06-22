# Clemson FCT Facial Recognition

This repository contains the code for facial recognition software that combines three popular algorithms PCA, LDA, and ICA.

## The Image Library

This software currently supports a subset of the [Netpbm](https://en.wikipedia.org/wiki/Netpbm_format) format, particularly with PGM and PPM images.

Images should __not__ be stored in this repository! Instead, images should be downloaded separately. Face databases are widely available on the Internet, such as [here](http://web.mit.edu/emeyers/www/face_databases.html) and [here](http://face-rec.org/databases/). I am currently using [this database](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

To convert JPEG images to PPM with ImageMagick:

```
for f in [images-folder]/**/*.jpg
do convert $f -size 300x200 "$(basename $f .jpg)".ppm
done
```

## The Matrix Library

Function Name              | PCA | LDA | ICA | Verification Status | Verify Date | Member
---                        |:---:|:---:|:---:|---                  |---          |---
_Constructors, Destructor_ |     |     |     |                     |             |
m_initialize               |  x  |  x  |  x  | Verified            |             |
m_identity                 |     |     |  x  | Verified            |             |
m_zeros                    |     |     |  x  | Verified            |             |
m_copy                     |     |     |  x  | Verified            |             |
m_free                     |  x  |  x  |  x  | Verified            |             |
_Input/Output_             |     |     |     |                     |             |
m_fprint                   |     |  x  |     | Verified            |             |
m_fwrite                   |  x  |     |     | Verified            |             |
m_fscan                    |     |     |     | Verified            |             |
m_fread                    |  x  |     |     | Verified            |             |
m_ppm_read                 |  x  |     |     | Verified            |             |
m_ppm_write                |  x  |     |     | Verified            |             |
_Getters_                  |     |     |     |                     |             |
m_eigenvalues_eigenvectors |  x  |  x  |  x  | Verified w/ LAPACK  |             |
m_matrix_multiply          |  x  |  x  |  x  | Verified w/ BLAS    |             |
m_mean_column              |  x  |     |     | Verified            |             |
m_transpose                |  x  |  x  |  x  | Verified            |             |
_Mutators_                 |     |     |     |                     |             |
m_normalize_columns        |  x  |     |     | Verified            |             |
_To review_                |     |     |     |                     |             |
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
m_findNonZeros             |     |     |     | Verified            | 10/06/2015  | James
m_reshape                  |     |     |     | Verified            | 10/06/2015  | James
m_inverseMatrix            |     |     |  x  | Not Verified        | 10/07/15    | Miller
m_norm                     |     |     |     | Not Verified        | 10/07/15    | Miller
m_sqrtm                    |     |     |  x  | Not Verified        | 10/07/15    | Miller
m_determinant              |     |     |     | Not Verified        | 10/07/15    | Miller
m_cofactor                 |     |     |     | Not Verified        | 10/07/15    | Miller
m_covariance               |     |     |  x  | Verified            | 11/05/15    | Greg
m_dot_subtract             |     |     |  x  | Verified            | 10/21/15    | Taylor
m_dot_add                  |     |     |  x  | Verified            | 10/21/15    | Taylor
m_dot_division             |     |     |  x  | Verified            | 11/03/15    | Greg
m_matrix_division          |     |     |     | Not Verified        | 10/21/15    | Taylor
m_reorder_columns          |     |     |  x  | Not Verified        | 10/21/15    | Taylor

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

    sudo apt-get install libblas-dev liblapacke-dev

##### Mac

Confirmed to run on Mac OS 10.11.1

Download LAPACK 3.5.0 http://www.netlib.org/lapack/

Download BLAS 3.5.0 http://www.netlib.org/blas/

(10.10 - 10.11) Download gfortran 5.2 http://coudert.name/software/gfortran-5.2-Yosemite.dmg

(10.7 - 10.9) Download gfortran 4.8.2 http://coudert.name/software/gfortran-4.8.2-MountainLion.dmg

    # in BLAS directory
    make
    sudo cp blas-LINUX.a /usr/local/lib/libblas.a

    # in LAPACK directory
    mv make.inc.example make.inc
    # set BLASLIB in make.inc line 68 equal to ‘/usr/local/lib/libblas.a’
    make
    sudo cp liblapack.a /usr/local/lib

## The Algorithms

Here is the working flow graph for the combined algorithm:

    m = number of dimensions per image
    n = number of images

    train: T -> (a, W', P)
        T = [T_1 ... T_n] (image matrix) (m-by-n)
        a = sum(T_i, 1:i:n) / n (mean face) (m-by-1)
        A = [(T_1 - a) ... (T_n - a)] (norm. image matrix) (m-by-n)
        W = pca(A), lda(A), ica(...) (projection matrix) (m-by-n)
        P = W' * A (projected images) (n-by-n)

    recognize: (a, W', P, T_i) -> P_match
        a = mean face (m-by-1)
        W = projection matrix (m-by-n)
        P = [P_1 ... P_n] (projected images) (n-by-n)
        T_i = test image (m-by-1)
        T_i_proj = W' * (T_i - a) (n-by-1)
        P_match = P_j that minimizes abs(P_j - T_i), 1:j:n (n-by-1)

    pca: A -> W_pca
        T = [T_1 ... T_n] (image matrix) (m-by-n)
        L = A' * A (surrogate matrix) (n-by-n)
        L_ev = eigenvectors of L (n-by-n)
        W_pca = A * L_ev (eigenfaces) (m-by-n)

    lda: P_pca -> W_opt
        S_b = (scatter around overall mean) (n-by-n)
        S_w = (scatter around mean of each class) (n-by-n)
        W_lda = c - 1 largest eigenvectors of S_w^-1 * S_b (m-by-(c - 1))
        W_pca = pca(A) (eigenfaces) (m-by-(n - c))
        W_opt' = W_lda' W_pca'

    ica: A -> W_ica
        ...

To run PCA on a training set of images:

    make
    ./train [training-images-folder]

To test a set of images against the training set:

    ./recognize [test-images-folder]
