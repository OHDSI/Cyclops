/*
 * Parameter.h
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace bsccs{


#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif


	class Parameter{
	public:
		Parameter(bsccs::real * data, int sizeIn);

		Parameter();

		~Parameter();

		int getSize();

		bsccs::real get(int index);

		bsccs::real getStored(int index);

		void initialize(bsccs::real * data, int sizeIn);


		void set(int index, bsccs::real setTo);

		void set(bsccs::real* newData);

		void logParameter();

		void logStored();

		void logParameter(const char * fileName);

		void store();

		void restore();

		bool getRestorable();

		void setRestorable(bool status);

		void resetChangesRecorded();

		std::vector<int> returnIndicesOfChanges();

		std::vector<double> returnCurrentValues();

		std::vector<double> returnStoredValues();

		std::vector<double> * returnCurrentValuesPointer();

		std::vector<double> * returnStoredValuesPointer();


	private:

		int size;

		bsccs::real * parameterValues;

		bsccs::real * storedValues;

		std::vector<double> parameterDoubleValues;

		std::vector<double> storedDoubleValues;

		std::vector<bool> vectorOfChanges;

		int numberOfChanges;

		bool restorable;
	};
}


#endif /* PARAMETER_H_ */
