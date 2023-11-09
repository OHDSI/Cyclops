/*
 * InputReader.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>


#include <string>

#include <iterator>

#include "InputReader.h"

#define MAX_ENTRIES		1000000000

#define FORMAT_MATCH_1	"CONDITION_CONCEPT_ID"
#define MATCH_LENGTH_1	20
#define FORMAT_MATCH_2	"PID"
#define MATCH_LENGTH_2	3

#define MISSING_STRING	"NA"

#ifdef USE_DRUG_STRING
	#define NO_DRUG			"0"
#else
	#define	NO_DRUG			0
#endif

namespace bsccs {

using namespace std;

//InputReader::InputReader() :
//    logger(bsccs::make_shared<loggers::CoutLogger>()),
//    error(bsccs::make_shared<loggers::CerrErrorHandler>()),
//    modelData(new ModelData(ModelType::NONE, logger, error)),
//    deleteModelData(true)
//{}

InputReader::InputReader(
	loggers::ProgressLoggerPtr _logger,
	loggers::ErrorHandlerPtr _error
) : logger(_logger), error(_error), modelData(new ModelData<double>(ModelType::NONE, _logger, _error)), deleteModelData(true) {
	// Do nothing
}

bool InputReader::listContains(const vector<IdType>& list, IdType value) {
	return (find(list.begin(), list.end(), value)
				!=  list.end());
}

InputReader::~InputReader() {
	if (deleteModelData) {
		delete modelData;
	}
}

} // namespace

