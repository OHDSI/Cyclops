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
	if (hXjX) {
		free(hXjX);
	}
}

void AbstractModelSpecifics::initialize(
		int iN,
		int iK,
		int iJ,
		CompressedDataMatrix* iXI,
		real* iNumerPid,
		real* iNumerPid2,
		real* iDenomPid,
		int* iNEvents,
		real* iXjY,
		std::vector<std::vector<int>* >* iSparseIndices,
		int* iPid,
		real* iOffsExpXBeta,
		real* iXBeta,
		int* iOffs,
		real* iBeta,
		real* iY,
		real* iWeights
		) {
	N = iN;
	K = iK;
	J = iJ;
	hXI = iXI;
	numerPid = iNumerPid;
	numerPid2 = iNumerPid2;
	denomPid = iDenomPid;
	hNEvents = iNEvents;

	sparseIndices = iSparseIndices;

	hPid = iPid;
	offsExpXBeta = iOffsExpXBeta;

	hXBeta = iXBeta;
	hOffs = iOffs;

	hBeta = iBeta;

	hY = iY;
	hWeights = iWeights;

//	hPid[100] = 0;  // Gets used elsewhere???
//	hPid[101] = 1;

	if (allocateXjY()) {
		hXjY = iXjY;
	}

	// TODO Should allocate host memory here

	hXjX = NULL;
	if (allocateXjX()) {
		hXjX = (real*) malloc(sizeof(real) * J);
	}
}
