/*
 * MarkovChainMonteCarlo.h
 *
 *  Created on: Aug 5, 2012
 *      Author: trevorshaddox
 */


#ifndef MARKOVCHAINMONTECARLO_H_
#define MARKOVCHAINMONTECARLO_H_


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


namespace BayesianSCCS {

class MarkovChainMonteCarlo {

public:

	MarkovChainMonteCarlo(int numberOfIterations, int vectorSize, gsl_vector * means, gsl_matrix * covarianceMatrix);

	virtual ~MarkovChainMonteCarlo();

	//NOT MY CODE - from gsl help line http://www.mail-archive.com/help-gsl@gnu.org/msg00631.html
	int rmvnorm(const gsl_rng *r, const int n, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);
	//NOT MY CODE - from gsl help line http://www.mail-archive.com/help-gsl@gnu.org/msg00631.html
	double dmvnorm(const int n, const gsl_vector *x, const gsl_vector *mean, const gsl_matrix *var);

protected:

	int N; //number of MCMC repetitions
	Eigen::MatrixXf BetaValues; //Matrix of Beta values from MCMC samples

};

}

#endif /* MARKOVCHAINMONTECARLO_H_ */
