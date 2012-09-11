/*
 * Parameter.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#include "Parameter.h"


using namespace std;
namespace bsccs{

	Parameter::Parameter(bsccs::real * data, int sizeIn){
		parameterValues = (bsccs::real*) calloc(sizeIn, sizeof(bsccs::real));
		memcpy(parameterValues, data, sizeof(bsccs::real)*sizeIn);
		size = sizeIn;
		didValueGetChanged = false;
		shouldBeChanged = false;
	}

	Parameter::~Parameter(){

		free(parameterValues);

	}

	int Parameter::getSize(){
		return size;
	}

	bsccs::real	 Parameter::get(int index){
		return parameterValues[index];
	}

	bsccs::real Parameter::getStored(int index){
		return storedValues[index];
	}

	void Parameter::set(int index, bsccs::real setTo){
		parameterValues[index] = setTo;
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

	bool Parameter::getChangeStatus() {
		return didValueGetChanged;
	}

	bool Parameter::getNeedToChangeStatus() {
		return shouldBeChanged;
	}

	void Parameter::setChangeStatus(bool status) {
		didValueGetChanged = status;
	}

	void Parameter::setNeedToChangeStatus(bool status) {
		shouldBeChanged = status;
	}

	std::vector<double> Parameter::returnCurrentValues() {
		std::vector<double> returnVector;

		for (int i = 0; i < size; i++) {
			returnVector.push_back((double) parameterValues[i]);
		}

		return returnVector;
	}

	std::vector<double> Parameter::returnStoredValues() {
		std::vector<double> returnVector;

		for (int i = 0; i < size; i++) {
			returnVector.push_back((double) storedValues[i]);
		}

		return returnVector;
	}

}



