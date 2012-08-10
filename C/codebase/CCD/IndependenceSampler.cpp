/*
 * IndependenceSampler.cpp
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include "IndependenceSampler.h"
#include "Eigen/core"


namespace bsccs {

IndependenceSampler::IndependenceSampler() {

}


IndependenceSampler::~IndependenceSampler() {

}

/***************************************************************************************
 *  Multivariate Normal density function and random number generator
 *  Multivariate Student t density function and random number generator
 *  Wishart random number generator
 *  Using GSL -> www.gnu.org/software/gsl
 *
 *  Copyright (C) 2006  Ralph dos Santos Silva
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  AUTHOR
 *     Ralph dos Santos Silva,  [EMAIL PROTECTED]
 *     March, 2006
***************************************************************************************/


/*****************************************************************************************************************/
/*****************************************************************************************************************/
int IndependenceSampler::rmvnorm(gsl_rng * r, const int n, gsl_vector *mean, gsl_matrix *var, gsl_vector *result){
/* multivariate normal distribution random number generator */
/*
*	n	dimension of the random vetor
*	mean	vector of means of size n
*	var	variance matrix of dimension n x n
*	result	output variable with a sigle random vector normal distribution generation
*/
int k;
gsl_matrix *work = gsl_matrix_alloc(n,n);

gsl_matrix_memcpy(work,var);
gsl_linalg_cholesky_decomp(work);

for(k=0; k<n; k++) {
	cout << "k = " << gsl_ran_ugaussian(r) << endl;
	gsl_vector_set(result, k, gsl_ran_ugaussian(r));
}
gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
gsl_vector_add(result,mean);

gsl_matrix_free(work);

return 0;
}

int IndependenceSampler::rmvnorm_stl(gsl_rng * r, const int n, gsl_vector *mean, gsl_matrix *var, gsl_vector *result) {
	int k;
	gsl_matrix *work = gsl_matrix_alloc(n,n);
	gsl_matrix_memcpy(work,var);
	gsl_linalg_cholesky_decomp(work);

	cout << "in RMVNORM: var = " << endl;
	for (int i = 0; i < n; i++) {
		cout << "[";
		for (int j = 0; j< n; j++) {
			cout << gsl_matrix_get(var, i, j) << ",";
		}
		cout << "]" << endl;
	}

	for(k=0; k<n; k++) {
		gsl_vector_set(result, k, gsl_ran_ugaussian(r));
	}
	gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
	gsl_vector_add(result,mean);

	gsl_matrix_free(work);

	return 0;


}

/*****************************************************************************************************************/
/*****************************************************************************************************************/
double IndependenceSampler::dmvnorm(const int n, gsl_vector *x, gsl_vector *mean, gsl_matrix *var){
/* multivariate normal density function    */
/*
*	n	dimension of the random vetor
*	mean	vector of means of size n
*	var	variance matrix of dimension n x n
*/
int s;
double ax,ay;
gsl_vector *ym, *xm;
gsl_matrix *work = gsl_matrix_alloc(n,n),
           *winv = gsl_matrix_alloc(n,n);
gsl_permutation *p = gsl_permutation_alloc(n);

gsl_matrix_memcpy( work, var );
gsl_linalg_LU_decomp( work, p, &s );
gsl_linalg_LU_invert( work, p, winv );
ax = gsl_linalg_LU_det( work, s );
gsl_matrix_free( work );
gsl_permutation_free( p );

xm = gsl_vector_alloc(n);
gsl_vector_memcpy( xm, x);
gsl_vector_sub( xm, mean );
ym = gsl_vector_alloc(n);
gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
gsl_matrix_free( winv );
gsl_blas_ddot( xm, ym, &ay);
gsl_vector_free(xm);
gsl_vector_free(ym);
ay = exp(-0.5*ay)/sqrt( pow((2*M_PI),n)*ax );

return ay;
}



}
