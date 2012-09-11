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

		~Parameter();

		int getSize();

		bsccs::real get(int index);

		bsccs::real getStored(int index);

		void set(int index, bsccs::real setTo);

		void logParameter();

		void logParameter(const char * fileName);

		void store();

		void restore();

		bool getChangeStatus();

		bool getNeedToChangeStatus();

		void setChangeStatus(bool status);

		void setNeedToChangeStatus(bool status);

		std::vector<double> returnCurrentValues();

		std::vector<double> returnStoredValues();

	private:

		int size;

		bsccs::real * parameterValues;

		bsccs::real * storedValues;

		bool didValueGetChanged;

		bool shouldBeChanged;

	};
}


#endif /* PARAMETER_H_ */
