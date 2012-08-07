/*
 * IndependenceSampler.h
 *
 *  Created on: Aug 6, 2012
 *      Author: trevorshaddox
 */

#ifndef INDEPENDENCESAMPLER_H_
#define INDEPENDENCESAMPLER_H_


#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

#include <Eigen/core>

#include <gsl/gsl_rng.h>


using std::cout;
using std::cerr;
using std::endl;
//using std::in;
using std::ifstream;


namespace bsccs {

class IndependenceSampler {

public:

	IndependenceSampler();

	virtual ~IndependenceSampler();

	int rmvnorm_stl(gsl_rng * r, const int n, gsl_vector *mean, gsl_matrix *var, gsl_vector *result);
	//NOT MY CODE - from gsl help line http://www.mail-archive.com/help-gsl@gnu.org/msg00631.html
	int rmvnorm(gsl_rng * r, const int n, gsl_vector *mean, gsl_matrix *var, gsl_vector *result);
	//NOT MY CODE - from gsl help line http://www.mail-archive.com/help-gsl@gnu.org/msg00631.html
	double dmvnorm(const int n, gsl_vector *x, gsl_vector *mean, gsl_matrix *var);

protected:


};

}

#endif /* INDEPENDENCESAMPLER_H_ */
