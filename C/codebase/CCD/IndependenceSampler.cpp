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


#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

namespace bsccs {

IndependenceSampler::IndependenceSampler() {

}


IndependenceSampler::~IndependenceSampler() {

}

void IndependenceSampler::sample(Parameter * Beta_Hat, Parameter * Beta,
		std::vector<std::vector<bsccs::real> > * cholesky, boost::mt19937& rng, double tuningParameter,
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
		//independentNormal.push_back(normalValue);
		//cout << "independent normal " << independentNormal[i] << endl;
		b[i] = exp(tuningParameter)*normalValue;
		//cout << "b[i] " << b[i] << endl;
	}

	/*
	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real actualValue = 0;
		for (int j = 0; j < sizeOfSample; j++) {
			actualValue += exp(tuningParameter)*cholesky[i][j]*independentNormal[j];
		}
		Beta->set(i, actualValue);// + Beta_Hat->get(i));
	}
	*/



	choleskyEigen.matrixL().solveInPlace(b);


	for (int i = 0; i < sizeOfSample; i++) {
		Beta->set(i, b[i] + Beta_Hat->get(i));
	}

}


}
