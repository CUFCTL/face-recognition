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
m_covariance               |     |     |  x  | Verified            |             |
m_eigenvalues_eigenvectors |  x  |  x  |  x  | Verified w/ LAPACK  |             |
m_inverse                  |     |  x  |  x  | Verified w/ LAPACK  |             |
m_mean_column              |  x  |  x  |     | Verified            |             |
m_product                  |  x  |  x  |  x  | Verified w/ BLAS    |             |
m_sqrtm                    |     |     |  x  | Verified            |             |
m_transpose                |  x  |  x  |  x  | Verified            |             |
_Mutators_                 |     |     |     |                     |             |
m_add                      |     |  x  |  x  | Verified            |             |
m_elem_mult                |     |  x  |  x  | Verified            |             |
m_subtract                 |     |  x  |  x  | Verified            |             |
m_subtract_columns         |  x  |     |     | Verified            |             |
_To review_                |     |     |     |                     |             |
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
        X = [(T_1 - a) ... (T_n - a)] (norm. image matrix) (m-by-n)
        W_pca' = PCA(X) (PCA projection matrix) (m-by-n)
        P_pca = W_pca' * X (PCA projected images) (n-by-n)
        W_lda' = LDA(W_pca, P_pca) (LDA projection matrix) (m-by-n)
        P_lda = W_lda' * X (LDA projected images) (n-by-n)
        W_ica' = ICA2(W_pca, P_pca) (ICA2 projection matrix) (n-by-n)
        P_ica = W_ica' * X (ICA2 projected images) (n-by-n)

    recognize: T_i -> P_match
        a = mean face (m-by-1)
        (W_pca, W_lda, W_ica) = projection matrices (m-by-n)
        (P_pca, P_lda, P_ica) = projected images (n-by-n)
        T_i = test image (m-by-1)
        P_test_pca = W_pca' * (T_i - a) (n-by-1)
        P_test_lda = W_lda' * (T_i - a) (n-by-1)
        P_test_ica = W_ica' * (T_i - a) (n-by-1)
        P_match_pca = nearest neighbor of P_test_pca (n-by-1)
        P_match_lda = nearest neighbor of P_test_lda (n-by-1)
        P_match_ica = nearest neighbor of P_test_ica (n-by-1)

    PCA: X -> W_pca'
        T = [T_1 ... T_n] (image matrix) (m-by-n)
        L = X' * X (surrogate matrix) (n-by-n)
        L_ev = eigenvectors of L (n-by-n)
        W_pca = X * L_ev (eigenfaces) (m-by-n)

    LDA: (W_pca, P_pca) -> W_lda'
        S_b = (scatter around overall mean) (n-by-n)
        S_w = (scatter around mean of each class) (n-by-n)
        W_fld = eigenvectors of S_w^-1 * S_b (n-by-n)
        W_lda' = W_fld' * W_pca' (n-by-m)

    ICA2: (W_pca, P_pca) -> W_ica'
        W_z = 2 * Cov(X)^(-1/2) (n-by-n)
        W = (train with sep96) (n-by-n)
        W_I = W * W_z (n-by-n)
        W_ica' = W_I * W_pca' (n-by-m)

To run PCA, LDA, and ICA2 on a training set of images:

    make
    ./train [training-images-folder]

To test a set of images against the training set:

    ./recognize [test-images-folder]
