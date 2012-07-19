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
//		int* iNEvents,
		real* iXjY,
		std::vector<std::vector<int>* >* iSparseIndices,
		int* iPid,
		real* iOffsExpXBeta,
		real* iXBeta,
		int* iOffs,
		real* iBeta,
		real* iY//,
//		real* iWeights
		) {
	N = iN;
	K = iK;
	J = iJ;
	hXI = iXI;
	numerPid = iNumerPid;
	numerPid2 = iNumerPid2;
	denomPid = iDenomPid;


	sparseIndices = iSparseIndices;

	hPid = iPid;
	offsExpXBeta = iOffsExpXBeta;

	hXBeta = iXBeta;
	hOffs = iOffs;

	hBeta = iBeta;

	hY = iY;
//	hKWeights = iWeights;

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

//#ifdef TRY_REAL
////	hNWeight.resize(N);
////	for (int i = 0; i < N; ++i) {
////		hNWeight[i] = static_cast<real>(iNEvents[i]);
////		cerr << iNEvents[i] << " " << hNWeight[i] << endl;
////	}
//#else
//	hNEvents = iNEvents;
//#endif

}
