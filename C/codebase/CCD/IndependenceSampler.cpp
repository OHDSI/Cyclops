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
#include <ctime>
#include <set>

#include <math.h>

#include "IndependenceSampler.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>


#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

namespace bsccs {

IndependenceSampler::IndependenceSampler() {

}


IndependenceSampler::~IndependenceSampler() {

}

void IndependenceSampler::sample(Parameter * Beta_Hat, Parameter * Beta,
		boost::mt19937& rng, double tuningParameter,
		Eigen::LLT<Eigen::MatrixXf> & choleskyEigen) {
	//TODO Better rng passing...  Make wrapper


	Beta->store();
	int sizeOfSample = Beta->getSize();


	vector<bsccs::real> independentNormal;  //Sampled independent normal values

	boost::normal_distribution<> nd(0.0, 1.0);

	boost::variate_generator<boost::mt19937&,
	                           boost::normal_distribution<> > var_nor(rng, nd);

	Eigen::VectorXf b = Eigen::VectorXf::Random(sizeOfSample);
	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real normalValue = var_nor();
		b[i] = exp(tuningParameter/2)*normalValue;

	}

	(choleskyEigen.matrixU()).solveInPlace(b);


	for (int i = 0; i < sizeOfSample; i++) {
		Beta->set(i, b[i] + Beta_Hat->get(i));
	}
	/*
	cout << "Printing Beta_Hat" << endl;
	Beta_Hat->logParameter();
	cout << "That was Beta_Hat" << endl;

	cout <<"PRINTING BETA" << endl;
	Beta->logParameter();
	cout <<"THAT WAS BETA" << endl;
	*/
}


}
