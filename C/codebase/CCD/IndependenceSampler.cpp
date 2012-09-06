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
#include "Eigen/core"

#define PI	3.14159265358979323851280895940618620443274267017841339111328125

namespace bsccs {

IndependenceSampler::IndependenceSampler() {

}


IndependenceSampler::~IndependenceSampler() {

}

void IndependenceSampler::sample(Parameter * Beta_Hat, Parameter * Beta, std::vector<std::vector<bsccs::real> > Cholesky_notGSL) {
	int sizeOfSample = Beta->getSize();

	vector<bsccs::real> independentNormal;  //Sampled independent normal values

	double unifRand1;
	double unifRand2;
	srand(time(NULL));
	for (int i = 0; i < sizeOfSample; i++) {
		unifRand1 = (rand() / (RAND_MAX + 1.0));
		unifRand2 = (rand() / (RAND_MAX + 1.0));
		bsccs::real normalValue = sqrt(-2*log(unifRand1)) * cos(2*PI*unifRand2); // Box-Muller method
		independentNormal.push_back(normalValue);
	}

	vector<bsccs::real> intermediateVector;

	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real current;
		current = independentNormal[i];
		for (int j = 0; j < sizeOfSample; j++) {
			current += Cholesky_notGSL[i][j]*Beta_Hat->get(j);
		}
		intermediateVector.push_back(current);
	}

	for (int i = 0; i < sizeOfSample; i++) {
		bsccs::real actualValue = 0;
		for (int j = 0; j < sizeOfSample; j++) {
			actualValue += Cholesky_notGSL[i][j]*intermediateVector[j];
		}
		Beta->set(i, actualValue);
	}

}


}
