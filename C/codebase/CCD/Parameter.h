/*
 * Parameter.h
 *
 *  Created on: Aug 10, 2012
 *      Author: trevorshaddox
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

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

		void set(int index);

		void logParameter();

		void logParameter(const char * fileName);

		void store();

		void restore();

	private:

		int size;

		bsccs::real * parameterValues;

		bsccs::real * storedValues;

	};
}


#endif /* PARAMETER_H_ */
