/*
 * AbstractModelSpecifics.cpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#include "AbstractModelSpecifics.h"
#include "io/InputReader.h"

namespace bsccs {

//AbstractModelSpecifics::AbstractModelSpecifics(
//		const std::vector<real>& y,
//		const std::vector<real>& z) : hY(y), hZ(z) {
//	// Do nothing
//}

AbstractModelSpecifics::AbstractModelSpecifics(const ModelData& input)
	: oY(input.getYVectorRef()), oZ(input.getZVectorRef()),
	  oPid(input.getPidVectorRef()),
	  hY(const_cast<real*>(oY.data())), hZ(const_cast<real*>(oZ.data())),
	  hPid(const_cast<int*>(oPid.data()))
	  {
	// Do nothing
}

AbstractModelSpecifics::~AbstractModelSpecifics() {
	if (hXjX) {
		free(hXjX);
	}
	for (HessianSparseMap::iterator it = hessianSparseCrossTerms.begin();
			it != hessianSparseCrossTerms.end(); ++it) {
		delete it->second;
	}
}

void AbstractModelSpecifics::makeDirty(void) {
	hessianCrossTerms.erase(hessianCrossTerms.begin(), hessianCrossTerms.end());

	for (HessianSparseMap::iterator it = hessianSparseCrossTerms.begin();
			it != hessianSparseCrossTerms.end(); ++it) {
		delete it->second;
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
		int* iPid_unused,
		real* iOffsExpXBeta,
		real* iXBeta,
		real* iOffs,
		real* iBeta,
		real* iY_unused//,
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

//	hPid = iPid;
	offsExpXBeta = iOffsExpXBeta;

	hXBeta = iXBeta;
	hOffs = iOffs;

//	hBeta = iBeta;

//	hY = iY;


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

} // namespace
