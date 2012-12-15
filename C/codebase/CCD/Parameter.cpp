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
		storedValues = (bsccs::real*) calloc(size, sizeof(bsccs::real));
		memcpy(storedValues, data, sizeof(bsccs::real)*sizeIn);
		memcpy(parameterValues, data, sizeof(bsccs::real)*sizeIn);
		size = sizeIn;
		didValueGetChanged = false;
		shouldBeChanged = false;

		for (int i = 0; i < size; i++) {
			parameterDoubleValues.push_back(1.00);
		}

		for (int i = 0; i < size; i++) {
			storedDoubleValues.push_back(1.00);
		}

	}

	Parameter::~Parameter(){
		free(storedValues);
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

	void Parameter::set(bsccs::real* newData){
		memcpy(parameterValues, newData, sizeof(bsccs::real)*size);
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
		memcpy(storedValues, parameterValues, sizeof(bsccs::real)*size);
	}

	void Parameter::restore(){

		bsccs::real* temp;
		temp = storedValues;
		storedValues = parameterValues;
		parameterValues = temp;

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

	std::vector<double> * Parameter::returnCurrentValuesPointer() {
		//std::vector<double> returnVector;

		for (int i = 0; i < size; i++) {
			//returnVector.push_back((double) parameterValues[i]);
			parameterDoubleValues[i] = (double) parameterValues[i];
		}


		return & parameterDoubleValues; //returnVector;
	}

	std::vector<double> * Parameter::returnStoredValuesPointer() {
		//std::vector<double> returnVector;

//		for (int i = 0; i < size; i++) {
//			//returnVector.push_back((double) storedValues[i]);
//			storedDoubleValues.push_back((double) storedValues[i]);
//		}

		for (int i = 0; i < size; i++) {
			storedDoubleValues[i] = (double) storedValues[i];
		}


		return & storedDoubleValues; //returnVector;
	}



}



