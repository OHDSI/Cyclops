/*
 * AbstractModelSpecifics.cpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#include "AbstractModelSpecifics.h"

AbstractModelSpecifics::AbstractModelSpecifics() {
	// Do nothing
}

AbstractModelSpecifics::~AbstractModelSpecifics() {
	// Do nothing
}

void AbstractModelSpecifics::initialize(
		int iN,
		int iK,
		CompressedDataMatrix* iXI,
		real* iNumerPid,
		real* iNumerPid2,
		real* iDenomPid,
		int* iNEvents,
		real* iXjEta,
		std::vector<std::vector<int>* >* iSparseIndices,
		int* iPid,
		real* iOffsExpXBeta,
		real* iXBeta,
		int* iOffs,
		real* iBeta,
		int* iEta,
		real* iWeights
		) {
	N = iN;
	K = iK;
	hXI = iXI;
	numerPid = iNumerPid;
	numerPid2 = iNumerPid2;
	denomPid = iDenomPid;
	hNEvents = iNEvents;
	hXjEta = iXjEta;
	sparseIndices = iSparseIndices;

	hPid = iPid;
	offsExpXBeta = iOffsExpXBeta;

	hXBeta = iXBeta;
	hOffs = iOffs;

	hBeta = iBeta;

	hEta = iEta;
	hWeights = iWeights;

//	hPid[100] = 0;  // Gets used elsewhere???
//	hPid[101] = 1;
}
