/*==================================================================================================
 *  runicaAux.c
 *
 *  Edited by William Halsey and Scott Rodgers
 *  whalsey@g.clemson.edu
 *  srodger@g.clemson.edu
 *
 *  This file contains
 *      spherex
 *      sep96
 *      sepout
 *      wchange
 *
 *  Lasted Edited: Jun. 20, 2013
 *
 *  Changes made: by William - changed all type "double" to type "data_t" as defined in
 *  matrix_manipulation.c
 *
 */

#include <stdio.h>
#include "matrix_manipulation.h"
/*==================================================================================================
 *  spherex
 *
 *  Parameters
 *	x/matrix, oldx,
 *  Returns
 *
 *  Description:
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *
 */
spherex(matrix_t *x, matrix_t *oldx, int rows, int cols, double P, matrix_t *wz) {
    matrix_t* mx = m_intialize(UNDEFINED, cols, rows);
	matrix_t* temp = m_intialize(UNDEFINED, P, 1);
	matrix_t* c = m_intialize(UNDEFINED, cols, rows);
	matrix_t* temp1 = m_intialize(UNDEFINED, cols, 1);
	matrix_t* transp = m_intialize(UNDEFINED, rows, cols);


	// allocate_matrix(&mx, cols, rows);
	// allocate_matrix(&temp, P, 1);
	// allocate_matrix(&c, cols, rows);
	// allocate_matrix(&temp1, cols, 1);
	// allocate_matrix(&transp, rows,cols);

	if (rows!=cols)
		fprintf(stderr,"spherex : rows!=cols. input matrix must be square  rows=%d,  cols=%d",rows,cols);
												//  % SPHEREX - spheres the training vector x.
                                                //  %    Requires x, P, to be predefined, and defines mx, c, wz.
                                                //
    printf("\nSubtracting Mean...\n");

	mean_of_matrix_by_rows(x,oldx,rows,cols);		//same as mean(transpose(x))
    ones(temp,P,1);
	// there is a problem here temp has p x 1 (deminsions) and mx has cols rows
    temp1 = m_multiply_matrices(temp, mx;
    x = m_dot_subtract(x,temp1);             //  x=x-(ones(P,1)*mx)';
    printf("Calculating whitening filter\n");
    //covariance(c, transp, rows, cols);          	            //  c=cov(x');
    c = m_covariance(transp)
	//matrix_sqrt(mx,c,rows);                      //  wz=2*inv(sqrtm(c));
    mx = m_sqrtm(c)
    //inv(c, mx, int rows);
    c = m_inverseMatrix(mx);
	//scale_matrix(wz, c, int rows, int cols, 2);

    wz = m_elem_mult (c, 2);

    printf("Whitening...\n");
    m_free(mx);
    mx = m_copy(x);
    x = m_multiply_matrices(wz, mx);
    //multiply_matrices(x, wz, mx);               //  x=wz*x;

    m_free(mx)
    m_free(temp);
    m_free(temp1);
    m_free(transp)
    //free_matrix(&mx);
    // free_matrix(&temp);
    // free_matrix(&c);
    // free_matrix(&temp1);
    // free_matrix(&transp);
    return;
}

/*==================================================================================================
 *  sep96
 *
 *  Parameters
 *
 *  Returns
 *
 *  Description: THINGS THAT NEED WORK
 *      figure out what "do_something" function should do (change name)
 *		write a negate function write a exp function
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *
 */
sep96(matrix_t *x, matrix_t *w, int *perm, int sweep, int count, int N, int M, int P, int B, int L, float angle, int change, matrix_t **ID) {
// % sep96.m implements the learning rule described in Bell \& Sejnowski, Vision
// % Research, in press for 1997, that contained the natural gradient (w'w).
// %
// % Bell & Sejnowski hold the patent for this learning rule.
// %
// % SEP goes once through the mixed signals, x
// % (which is of length M), in batch blocks of size B, adjusting weights,
// % w, at the end of each block.
// % sepout is called every F counts.
// %
// % I suggest a learning rate (lrate) of 0.006, and a blocksize (B) of
// % 300, at least for 2->2 separation.
// % When annealing to the right solution for 10->10, however, lrate of
// % less than 0.0001 and B of 10 were most successful.
// %
// % Copyright 1996 Tony Bell
// % This may be copied for personal or academic use.
// % For commercial use, please contact Tony Bell
// % (tony@salk.edu) for a commercial license.
	matrix_t* BI = m_intialize(UNDEFINED, rows, rows);
	matrix_t* temp_u = m_intialize(UNDEFINED, rows, rows);
	matrix_t* temp_u1 = m_intialize(UNDEFINED, EIGV_NUM, num_pixels);
    matrix_t* colon_matrix = m_intialize(rows, rows, num_pixels);
	matrix_t* u_transposed = m_intialize(UNDEFINED, cols, rows);


// 	//allocate_matrix(&BI, ?rows, ?rows);
//   allocate_matrix(&BI, rows, rows); //ZH we believe its rows by rows, BI = B * ID

// //	allocate_matrix(&temp_u, ?, ?);
//   allocate_matrix(&temp_u, rows, rows); // ZH

// //  allocate_matrix(&temp_u1, ?, ?);
// //	allocate_matrix(&u_transposed, ?cols, ?rows);


//   allocate_matrix(&temp_u1, rows, rows); // ZH
// 	allocate_matrix(&u_transposed, cols, rows); //ZH
  // x=x(:,perm);
    x = m_reorder_columns(x,perm)
    sweep=sweep+1;
    int t=1;             // sweep=sweep+1; t=1;

    // noblocks=fix(P/B);
    noblocks = (int) ((double) P / B);

    BI = m_elem_mult(ID, B);
    scale_matrix(BI, ID, ?rows, ?, B);  // BI=B*Id;
    BI = m_multiply_matrices(ID,B);

    // for t=t:B:t + noblocks * B - 1
    int i, j;
    for ( i = t; i = t - 1 + noblocks * B; i += B ) {
        // count = count + B;
        count=count+B;

		//allocate_matrix(&colon_matrix, ?rows, ?rows);

	    // u = w * x(:, t:t + B - 1);
        temp_u = m_product(w, x);

        // w = w + L * (BI + (1 - 2 * (1. / (1 + exp(-u)))) * u') * w;
        for ( j = 0; j < u->cols; j++ ) {
            elem(temp_u, j, 0) = 1 - 2 * (1 / (1 + exp(-elem(temp_u, j, 0))));
        }

        u_transposed = m_transpose(u)

        temp_u1 = m_product(temp_u, u_transposed);

		//is BI a scalar? // Next step is BI + temp_u1
		add_matrices(temp_u, temp_u1, BI, ?rows, ?cols) ;

        temp_u = m_dot_add(temp_u1, BI);

		//then W*L*(BI+temp_u1)
		//multiply_matrices(temp_u1, temp_u, L, ?rows, ?cols, ?cols);

        temp_u1 = m_multiply_matrices(temp_u, L);

		//multiply_matrices(temp_u, temp_u1, W, ?rows, ?cols, ?cols);

        temp_u = m_multiply_matrices(temp_u1, W);

		//then w = w+(W*L*(BI+temp_u1)
		//add_matrices(w, w, temp_u, ?rows, ?cols);

        w = m_dot_add(w,temp_u);


        if(count > f) {                                //   if count>F, sepout; count=count-F; end;
            sepout(oldw, w, olddelta, sweep, N, M, P, B, L, change, angle);
            count = count - F;
        }
        /*
         t =
         */
    }
	m_free(temp_u);
	m_free(temp_u1);

    return;
}

/*==================================================================================================
 *  sepout
 *
 *  Parameters
 *
 *  Returns
 *
 *  Description:
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *
 */
sepout(matrix_t *oldw, matrix_t *w, double *olddelta, int sweep, int N, int M, int P, int B, data_t L, int change, int angle) {
    int i, j;
	// is this needed anywhere else
	matrix_t* detla = m_intialize(UNDEFINED, rows, cols);
	//allocate_matrix(&delta, rows?, cols?);

    wchange(change, delta, angle, oldw, w, olddelta, M, N);    // [change,olddelta,angle]=wchange(oldw,w,olddelta);

    for(i = 0; i < rows of w; i++) {
        for (j = 0; j < cols of w; j++) oldw = w;   // oldw=w;
    }

    printf("****sweep=%d, change=%.4f angle=%.1f deg., [N%d,M%d,P%d,B%d,L%.5f]\n", sweep,change,180*angle/M_PI,N,M,P,B,L);   // fprintf('****sweep=%d, change=%.4f angle=%.1f deg., [N%d,M%d,P%d,B%d,L%.5f]\n', sweep,change,180*angle/pi,N,M,P,B,L);

    return;
}

/*==================================================================================================
 *  wchange
 *
 *  Parameters
 *      pointer, type double        = change
 *      double pointer, type data_t = delta
 *      pointer, type double        = angle
 *      double pointer, type data_t = w
 *      value, type integer         = wRows
 *      value, type integer         = wCols
 *      double pointer, type data_t = oldw
 *      double pointer, type data_t = olddelta
 *
 *  Returns
 *      N/A
 *      Implicitly returns values through variables "change," "delta," and "angle."
 *
 *  Description:
 *
 *  THIS FUNCTION CALLS
 *
 *  THIS FUNCTION IS CALLED BY
 *
 */
wchange(matrix_t *change, matrix_t *delta, double *angle, data_t *w, int wRows, int wCols,
    matrix_t *oldw, matrix_t *olddelta) {
                                                        //  % Calculates stats on most recent weight change - magnitude and angle between
                                                        //  % old and new weight.
                                                        //
                                                        //  function [change,delta,angle]=wchange(w,oldw,olddelta)
    matrix_t *tempA = m_intialize(UNDEFINED, deltaLen, 1);
    matrix_t *tempB = m_intialize(UNDEFINED, 1, 1);
    matrix_t *tempC = m_intialize(UNDEFINED, 1, 1);
    matrix_t *tdelta = m_intialize(UNDEFINED, rows, cols);
    matrix_t *oldw_w = m_intialize(UNDEFINED, rows, cols);

    int deltaLen = wRows * wCols;

    // allocate_matrix(&tempA, deltaLen);
 //    allocate_matrix(&tempB, 1, 1);
 //    allocate_matrix(&tempC, 1, 1);
	// allocate_matrix(&tdelta, rows?, cols?);
	// allocate_matrix(&oldw_w, rows?, cols?);

    oldw_w = m_dot_subtract(oldw, w);
    reshape(delta, 1, deltaLen, oldw_w, wRows, wCols);  //   [M,N]=size(w); delta=reshape(oldw-w,1,M*N);

    tdelta = m_transpose(delta);
    tempB = m_multiply_matrices(delta, tdelta);
    //multiply_matrices(tempB, delta, tdelta, 1, 1, deltaLen);
    *change = tempB[0];                              //   change=delta*delta';

	/*  olddelta'   */
    tempA = m_transpose(olddelta);
    /*  delta*olddelta' */
    tempB = m_ multiply_matrices(delta, tempA);
    /*  olddelta*olddelta'  */
    //multiply_matrices(tempC, olddelta, tempA, 1, 1, deltaLen);

    tempC = m_ multiply_matrices(olddelta, tempA);

    /*  sqrt((delta*delta')*(olddelta*olddelta'))   */

    double tempD = sqrt(change[0] * tempC[0]);

    //matrix_t *tempD = m_intialize(UNDEFINED, sqrt(change[0] * tempC[0]), 1);

    /*  (delta*olddelta')/sqrt((delta*delta')*(olddelta*olddelta')) */
    tempD = (tempB[0]) / tempD;
    *angle = acos(tempD);

	// old code don't delete yet
    /*  olddelta'   */
    // transpose(tempA, olddelta, 1, deltaLen);
    /*  delta*olddelta' */
    // multiply_matrices(tempB, delta, tempA, 1, 1, deltaLen);
    /*  olddelta*olddelta'  */
    // multiply_matrices(tempC, olddelta, tempA, 1, 1, deltaLen);
    /*  sqrt((delta*delta')*(olddelta*olddelta'))   */
    // tempD = sqrt(change[0][0] * tempC[0][0]);
    /*  (delta*olddelta')/sqrt((delta*delta')*(olddelta*olddelta')) */
    // tempD = (tempB[0][0]) / tempD;
    // *angle = acos(tempD);                               //  angle=acos((delta*olddelta')/sqrt((delta*delta')*(olddelta*olddelta')));

    return;
}
