/*
 * Types.h
 *
 *  Created on: Aug 20, 2013
 *      Author: msuchard
 */

#ifndef CCD_TYPES_H_
#define CCD_TYPES_H_

namespace bsccs {

namespace Models {

enum ModelType {
	NONE = 0,
	NORMAL,
	POISSON,
	LOGISTIC,
	CONDITIONAL_LOGISTIC,
	SELF_CONTROLLED_MODEL,
	COX
};

//const static long RequiresStratumID = 1 << 0;
//const static long RequiresCensoredData = 1 << 1;
//const static long RequiresCensoredData2 = 1 << 2;
//
//const static long NormalRequirements = 0;
//const static long PoissonRequirements = 0;

static bool requiresStratumID(const ModelType modelType) {
	return (modelType == CONDITIONAL_LOGISTIC || modelType == SELF_CONTROLLED_MODEL);
}

static bool requiresCensoredData(const ModelType modelType) {
	return (modelType == COX);
}

static bool requiresOffset(const ModelType modelType) {
	return (modelType == SELF_CONTROLLED_MODEL);
}

//template<ModelType model>
//long getRequirements();
//
//template<ModelType NORMAL>
//long getRequirements() { return 0; }
//
}

}

#endif /* CCD_TYPES_H_ */
