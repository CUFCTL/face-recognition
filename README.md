# Clemson FCT Facial Recognition

## Library Functions

Function Name              |PCA  |LDA  |ICA  |Verification Status  |Verify Date|Member|Dataset|Last Edit
---                        |:---:|:---:|:---:|---                  |---        |---   |---    |---
m_initialize               |x||| Verified    | 10/21/15   | Taylor
m_free                     |x||| Verified    | 10/21/15   | Taylor
m_fprint                   | ||| Verified    | 10/21/15   | Taylor
m_fwrite                   |x||| Verified    | 10/21/15   | Taylor
m_fscan                    | ||| Verified    | 10/21/15   | Taylor
m_fread                    |x||| Verified    | 10/21/15   | Taylor
m_copy                     | ||| Verified    | 10/21/15   | Taylor
                           | |||             |            |
m_flipCols                 | ||| Verified    | 10/02/15   | James
m_normalize                | ||| Verified    | 10/02/2015 | James
m_elem_mult                | ||| Verified    | 10/02/2015 | James
m_elem_truncate            | ||| Verified    | 10/02/2015 | James
m_elem_divideByConst       | ||| Verified    | 10/02/2015 | James
m_elem_acos                | ||| Verified    | 10/02/2015 | James
m_elem_sqrt                | ||| Verified    | 10/02/2015 | James
m_elem_negate              | ||| Verified    | 10/02/2015 | James
m_elem_exp                 | ||| Verified    | 10/02/2015 | James
m_elem_pow                 | ||| Verified    | 10/02/2015 | James
m_elem_divideByMatrix      | ||| Verified    | 10/02/2015 | James
m_elem_add                 | ||| Verified    | 10/02/2015 | James
m_sumCols                  | ||| Verified    | 10/02/2015 | James
m_meanCols                 | ||| Verified    | 10/02/2015 | James
m_sumRows                  | ||| Verified    | 10/02/2015 | James
m_meanRows                 |x||| Verified    | 10/06/2015 | James
m_findNonZeros             | ||| Verified    | 10/06/2015 | James
m_transpose                |x||| Verified    | 10/06/2015 | James
m_reshape                  | ||| Verified    | 10/06/2015 | James
                           | |||             |            |
m_inverseMatrix            |x||| Not Verified| 10/07/15   | Miller
m_norm                     | ||| Not Verified| 10/07/15   | Miller
m_sqrtm                    | ||| Not Verified| 10/07/15   | Miller
m_determinant              | ||| Not Verified| 10/07/15   | Miller
m_cofactor                 | ||| Not Verified| 10/07/15   | Miller
m_covariance               | ||| Verified    | 11/05/15   | Greg
                           | |||             |            |
m_dot_subtract             | ||| Verified    | 10/21/15   | Taylor
m_dot_add                  | ||| Verified    | 10/21/15   | Taylor
m_dot_division             | ||| Verified    | 11/03/15   | Greg
m_matrix_multiply          |x||| Verified    | 10/21/15   | Taylor
m_matrix_division          | ||| Not Verified| 10/21/15   | Taylor
m_reorder_columns          | ||| Not Verified| 10/21/15   | Taylor
                           | |||             |            |
m_eigenvalues_eigenvectors |x||| Not Verified| 10/22/15   | Colin
m_getSubMatrix             | ||| Not Verified| 10/22/15   | Colin
loadPPMtoMatrixCol         |x||| Not Verified| 10/22/15   | Colin
writePPMgrayscale          |x||| Not Verified| 10/22/15   | Colin

_Note: When we say verified we are talking about an initial verification step which does not mean the function is fully trustworthy in the final code. There will be an additional possibility for this column: "Verified using BLAS" which will mean that the library function has be implemented in blas and is ready to be used in the final code_

##### Blas Libraries
| Much of the code in this project depends on blas. In order to run it properly, it is necessary to install the following blas and lapack libraries: |
| ---                             |
| libblas-dev (1.2.20110419-5)    |
| libblas3 (1.2.20110419-5)       |
| libgfortran3 (4.8.1-10ubuntu9)  |
| liblapack-dev (3.4.2+dfsg-2)    |
| liblapack3 (3.4.2+dfsg-2)       |
| liblapacke (3.4.2+dfsg-2)       |
| liblapacke-dev (3.4.2+dfsg-2)   |

This list of libraries was updated 11/4/2015 by Miller

##### Using PCA

To convert JPEG images to PPM, use `ffmpeg` in the containing directory:
```
for f in *.jpg
do ffmpeg -i $f -s 300x200 "$(basename $f .jpg)".ppm
done
```

The training set should be located as follows:

    PCA/
        pcaCreateDatabase
        pcaRecognition
    test_images/
        *.ppm

To use the training set:
```
./pcaCreateDatabase
```

To use a test set:
```
./pcaRecognition [input-file]
```
where input-file is a text file that lists the test set images
