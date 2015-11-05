#Clemson FCT Facial Recognition

##Team Members

####Miller Hall
- Current Project:
 - Assess progress of matrix operations library
 - Ensure all functions are working
 - Use blas to optimize the remaining functions
- Progress:
 - Assess progress of matrix operations library
- Next Step:
 - By the end of the semester all functions using blas
 - All code from project documented and in same repository
 - Improve code legibility
 - Move all helper functions into the central library
- Date Updated:
 - 11/5/2015

####Taylor Sieling
- Current Project:
 - Shared Code Team
- Progress:
 - Working on documentation and function verification. Also working on getting Blas working on my machine.
- Next Step:
 - Blas integration into the matrix library
- Date Updated:
 - 10/16/2015

####Colin Targonski
- Current Project:
 - Shared Code Team
- Progress:
 - Working on documentation and function verification. Also beginning work on BLAS library
- Next Step:
 - BLAS integration
- Date Updated:
 - 10/16/2015

####James Peterkin II
- Current Project:
 - Shared Code Team
- Progress:
 - Documenting working functions in group2 matrix functions
- Next Step:
 - Look at blas documentation
- Date Updated:
 - 10/16/2015

####Greg FitzMaurice
- Current Project:
 - Algorithms: PCA
- Progress:
 - Determine if actually complete.
- Next Step:
 - Test PCA functionality using semi-completed/completed shared library
- Date Updated:
 - 10/21/2015

####Zhong Hu
- Current Project:
 - Algorithms: ICA
- Progress:
 - Finishing few functions.
- Next Step:
 - Swap the matrix function with the functions in the shared library
- Date Updated:
 - 10/27/2015


##Algorithms

Algorithm                  | Verification Status        | Date       | Member
---                        | ---                        | ---        | ---

##Library Functions

Function Name              | Verification Status        |  Date      | Member
---                        | ---                        |  ---       | ---
m_initialize               | Verified                   | 10/21/15   | Taylor
m_free                     | Verified                   | 10/21/15   | Taylor
m_fprint                   | Verified                   | 10/21/15   | Taylor
m_fwrite                   | Verified                   | 10/21/15   | Taylor
m_fscan                    | Verified                   | 10/21/15   | Taylor
m_fread                    | Verified                   | 10/21/15   | Taylor
m_copy                     | Verified                   | 10/21/15   | Taylor
                           |                            |            | 
m_flipCols                 | Verified                   | 10/02/15   | James
m_normalize                | Verified                   | 10/02/2015 | James
m_elem_mult                | Verified                   | 10/02/2015 | James
m_elem_truncate            | Verified                   | 10/02/2015 | James
m_elem_divideByConst       | Verified                   | 10/02/2015 | James
m_elem_acos                | Verified                   | 10/02/2015 | James
m_elem_sqrt                | Verified                   | 10/02/2015 | James
m_elem_negate              | Verified                   | 10/02/2015 | James
m_elem_exp                 | Verified                   | 10/02/2015 | James
m_elem_pow                 | Verified                   | 10/02/2015 | James
m_elem_divideByMatrix      | Verified                   | 10/02/2015 | James
m_elem_add                 | Verified                   | 10/02/2015 | James
m_sumCols                  | Verified                   | 10/02/2015 | James
m_meanCols                 | Verified                   | 10/02/2015 | James
m_sumRows                  | Verified                   | 10/02/2015 | James
m_meanRows                 | Verified                   | 10/06/2015 | James
m_findNonZeros             | Verified                   | 10/06/2015 | James
m_transpose                | Verified                   | 10/06/2015 | James
m_reshape                  | Verified                   | 10/06/2015 | James
                           |                            |            |
m_inverseMatrix            | Not Verified               | 10/07/15   | Miller
m_norm                     | Not Verified               | 10/07/15   | Miller
m_sqrtm                    | Not Verified               | 10/07/15   | Miller
m_determinant              | Not Verified               | 10/07/15   | Miller
m_cofactor                 | Not Verified               | 10/07/15   | Miller
m_covariance               | Not Verified               | 10/07/15   | Miller
                           |                            |            |
m_dot_subtract             | Verified                   | 10/21/15   | Taylor
m_dot_add                  | Verified                   | 10/21/15   | Taylor
m_dot_division             | Not Verified               | 10/21/15   | Taylor
m_matrix_multiply          | Verified                   | 10/21/15   | Taylor
m_matrix_division          | Not Verified               | 10/21/15   | Taylor
m_reorder_columns          | Not Verified               | 10/21/15   | Taylor
                           |                            |            |
m_eigenvalues_eigenvectors | Not Verified               | 10/22/15   | Colin
m_getSubMatrix             | Not Verified               | 10/22/15   | Colin
loadPPMtoMatrixCol         | Not Verified               | 10/22/15   | Colin
writePPMgrayscale          | Not Verified               | 10/22/15   | Colin

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
