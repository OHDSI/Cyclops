/*
 * InputReader.h
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#ifndef INPUTREADER_H_
#define INPUTREADER_H_

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <map>

using std::map;
using std::string;
using std::vector;
using std::stringstream;

#ifdef MY_RCPP_FLAG
	#include <R.h>
//// For OSX 10.6, R is built with 4.2.1 which has a bug in stringstream
//stringstream& operator>> (stringstream &in, int &out) {
//	string entry;
//	in >> entry;
//	out = atoi(entry.c_str());
//	return in;
//}
#endif

#include "ModelData.h"
#include "io/ProgressLogger.h"

namespace bsccs {

    template <typename RealType>
class InputReader {
public:
//     InputReader();

//	InputReader(
//		loggers::ProgressLoggerPtr _logger,
//	    loggers::ErrorHandlerPtr _error
//	);

        InputReader(
                loggers::ProgressLoggerPtr _logger,
                loggers::ErrorHandlerPtr _error
        ) : logger(_logger), error(_error), modelData(new ModelData<RealType>(ModelType::NONE, _logger, _error)), deleteModelData(true) {
            // Do nothing
        }

	virtual ~InputReader() {
            if (deleteModelData) {
                delete modelData;
            }
        }

        virtual void readFile(const char* fileName) = 0;

	AbstractModelData* getModelData() {
		// TODO Use smart pointer
		deleteModelData = false;
		return modelData;
	}

protected:
//	bool listContains(const vector<IdType>& list, IdType value);

        bool listContains(const vector<IdType>& list, IdType value) {
            return (find(list.begin(), list.end(), value)
                    !=  list.end());
        }



	void split( vector<string> & theStringVector,  /* Altered/returned value */
	       const  string  & theString,
	       const  string  & theDelimiter) {
	//	istringstream iss(theString);
	//	copy(istream_iterator<string>(iss),
	//	         istream_iterator<string>(),
	//	         back_inserter<vector<string> >(theStringVector));
		tokenize(theString, theStringVector, theDelimiter);
	}

	template < class ContainerT >
	void tokenize(const std::string& str, ContainerT& tokens,
	              const std::string& delimiters = " ", const bool trimEmpty = false)
	{

		typedef ContainerT Base;
		typedef typename Base::value_type ValueType;
		typedef typename ValueType::size_type SizeType;
	   std::string::size_type pos, lastPos = 0;
	   while(true)
	   {
	      pos = str.find_first_of(delimiters, lastPos);
	      if(pos == std::string::npos)
	      {
	         pos = str.length();

	         if(pos != lastPos || !trimEmpty)
	            tokens.push_back(ValueType(str.data()+lastPos,
	                  (SizeType)pos-lastPos ));

	         break;
	      }
	      else
	      {
	         if(pos != lastPos || !trimEmpty)
	            tokens.push_back(ValueType(str.data()+lastPos,
	                  (SizeType)pos-lastPos ));
	      }

	      lastPos = pos + 1;
	   }
	}

	loggers::ProgressLoggerPtr logger;
	loggers::ErrorHandlerPtr error;

	ModelData<RealType>* modelData;
	bool deleteModelData;
};

} // namespace

#endif /* INPUTREADER_H_ */
