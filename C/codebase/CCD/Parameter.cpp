/*
 * Parameter.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#include "Parameter.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>

using namespace std;
namespace bsccs{

	Parameter::Parameter(bsccs::real * data, int sizeIn){
		parameterValues = (bsccs::real*) calloc(sizeIn, sizeof(bsccs::real));
		memcpy(parameterValues, data, sizeof(bsccs::real)*sizeIn);
		size = sizeIn;
	}

	Parameter::~Parameter(){

		free(parameterValues);

	}

	int Parameter::getSize(){
		return size;
	}

	bsccs::real Parameter::get(int index){
		return parameterValues[index];
	}

	void Parameter::logParameter() {
		cout << "Parameter value is <";
		for (int i = 0; i < size; i++) {
			cout << parameterValues[i] << ",";
		}
		cout << ">" << endl;

	}

	void Parameter::logParameter(const char * fileName) {
		// TODO Implement a write to file function...
	}

	void Parameter::store(){
		storedValues = (bsccs::real*) calloc(size, sizeof(bsccs::real));
		memcpy(storedValues, parameterValues, sizeof(bsccs::real)*size);
	}

	void Parameter::restore(){
		free(parameterValues);
		parameterValues = storedValues;
	}


}



