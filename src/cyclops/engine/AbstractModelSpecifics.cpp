/*
 * AbstractModelSpecifics.cpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#include <stdexcept>
#include <set>

#include "AbstractModelSpecifics.h"
#include "ModelData.h"
#include "engine/ModelSpecifics.h"
// #include "io/InputReader.h"

//#include "Rcpp.h"

namespace bsccs {

//bsccs::shared_ptr<AbstractModelSpecifics> AbstractModelSpecifics::factory(const ModelType modelType, const ModelData& modelData) {
//	bsccs::shared_ptr<AbstractModelSpecifics> model;
// 	switch (modelType) {
// 		case ModelType::SELF_CONTROLLED_MODEL :
// 			model =  bsccs::make_shared<ModelSpecifics<SelfControlledCaseSeries<real>,real> >(modelData);
// 			break;
// 		case ModelType::CONDITIONAL_LOGISTIC :
// 			model =  bsccs::make_shared<ModelSpecifics<ConditionalLogisticRegression<real>,real> >(modelData);
// 			break;
// 		case ModelType::LOGISTIC :
// 			model = bsccs::make_shared<ModelSpecifics<LogisticRegression<real>,real> >(modelData);
// 			break;
// 		case ModelType::NORMAL :
// 			model = bsccs::make_shared<ModelSpecifics<LeastSquares<real>,real> >(modelData);
// 			break;
// 		case ModelType::POISSON :
// 			model = bsccs::make_shared<ModelSpecifics<PoissonRegression<real>,real> >(modelData);
// 			break;
//		case ModelType::CONDITIONAL_POISSON :
// 			model = bsccs::make_shared<ModelSpecifics<ConditionalPoissonRegression<real>,real> >(modelData);
// 			break;
// 		case ModelType::COX_RAW :
// 			model = bsccs::make_shared<ModelSpecifics<CoxProportionalHazards<real>,real> >(modelData);
// 			break;
// 		case ModelType::COX :
// 			model = bsccs::make_shared<ModelSpecifics<BreslowTiedCoxProportionalHazards<real>,real> >(modelData);
// 			break;
// 		default:
// 			throw std::invalid_argument("Unknown modelType");
// 			break;
// 	}
//	return model;
//}

AbstractModelSpecifics* AbstractModelSpecifics::factory(const ModelType modelType, const ModelData& modelData) {
	AbstractModelSpecifics* model = nullptr;
 	switch (modelType) {
 		case ModelType::SELF_CONTROLLED_MODEL :
 			model =  new ModelSpecifics<SelfControlledCaseSeries<real>,real>(modelData);
 			break;
 		case ModelType::CONDITIONAL_LOGISTIC :
 			model =  new ModelSpecifics<ConditionalLogisticRegression<real>,real>(modelData);
 			break;
 		case ModelType::TIED_CONDITIONAL_LOGISTIC :
 			model =  new ModelSpecifics<TiedConditionalLogisticRegression<real>,real>(modelData);
 			break;
 		case ModelType::LOGISTIC :
 			model = new ModelSpecifics<LogisticRegression<real>,real>(modelData);
 			break;
 		case ModelType::NORMAL :
 			model = new ModelSpecifics<LeastSquares<real>,real>(modelData);
 			break;
 		case ModelType::POISSON :
 			model = new ModelSpecifics<PoissonRegression<real>,real>(modelData);
 			break;
		case ModelType::CONDITIONAL_POISSON :
 			model = new ModelSpecifics<ConditionalPoissonRegression<real>,real>(modelData);
 			break;
 		case ModelType::COX_RAW :
 			model = new ModelSpecifics<CoxProportionalHazards<real>,real>(modelData);
 			break;
 		case ModelType::COX :
 			model = new ModelSpecifics<BreslowTiedCoxProportionalHazards<real>,real>(modelData);
 			break;
 		default:
 			break;
 	}
	return model;
}

//AbstractModelSpecifics::AbstractModelSpecifics(
//		const std::vector<real>& y,
//		const std::vector<real>& z) : hY(y), hZ(z) {
//	// Do nothing
//}

AbstractModelSpecifics::AbstractModelSpecifics(const ModelData& input)
	: //oY(input.getYVectorRef()), oZ(input.getZVectorRef()),
	  //oPid(input.getPidVectorRef()),
	  modelData(input),
//	  hXI(static_cast<CompressedDataMatrix*>(const_cast<ModelData*>(&modelData))),
// 	  hY(const_cast<real*>(input.getYVectorRef().data())), //hZ(const_cast<real*>(input.getZVectorRef().data())),
	  hY(input.getYVectorRef()),
// 	  hOffs(const_cast<real*>(input.getTimeVectorRef().data())),
	  hOffs(input.getTimeVectorRef()),
// 	  hPid(const_cast<int*>(input.getPidVectorRef().data()))
// 	  hPid(input.getPidVectorRef())
      hPidOriginal(input.getPidVectorRef()), hPid(const_cast<int*>(hPidOriginal.data()))
	  {
	// Do nothing
}

AbstractModelSpecifics::~AbstractModelSpecifics() { }

void AbstractModelSpecifics::makeDirty(void) {
	hessianCrossTerms.erase(hessianCrossTerms.begin(), hessianCrossTerms.end());

//	for (HessianSparseMap::iterator it = hessianSparseCrossTerms.begin();
//			it != hessianSparseCrossTerms.end(); ++it) {
//		delete it->second;
//	}
}

int AbstractModelSpecifics::getAlignedLength(int N) {
	return (N / 16) * 16 + (N % 16 == 0 ? 0 : 16);
}


void AbstractModelSpecifics::setPidForAccumulation(const real* weights) {

	hPidInternal =  hPidOriginal; // Make copy
	hPid = hPidInternal.data(); // Point to copy
	accReset.clear();

	const int ignore = -1;

	// Find first non-zero weight
	size_t index = 0;
	while(weights != nullptr && weights[index] == 0.0 && index < K) {
		hPid[index] = ignore;
		index++;
	}

	int lastPid = hPid[index];
	real lastTime = hOffs[index];
	real lastEvent = hY[index];

	int pid = hPid[index] = 0;

	for (size_t k = index + 1; k < K; ++k) {
		if (weights == nullptr || weights[k] != 0.0) {
			int nextPid = hPid[k];

			if (nextPid != lastPid) { // start new strata
				pid++;
				accReset.push_back(pid);
				lastPid = nextPid;
			} else {

				if (lastEvent == 1.0 && lastTime == hOffs[k] && lastEvent == hY[k]) {
					// In a tie, do not increment denominator
				} else {
					pid++;
				}
			}
			lastTime = hOffs[k];
			lastEvent = hY[k];

			hPid[k] = pid;
		} else {
			hPid[k] = ignore;
		}
	}
	pid++;
	accReset.push_back(pid);

	// Save number of denominators
	N = pid;

	if (weights != nullptr) {
		for (size_t i = 0; i < K; ++i) {
			if (hPid[i] == ignore) hPid[i] = N; // do NOT accumulate, since loops use: i < N
		}
	}
	setupSparseIndices(N); // ignore pid == N (pointing to removed data strata)
}

void AbstractModelSpecifics::setupSparseIndices(const int max) {
	sparseIndices.clear(); // empty if full!

	for (size_t j = 0; j < J; ++j) {
		if (modelData.getFormatType(j) == DENSE || modelData.getFormatType(j) == INTERCEPT) {
			sparseIndices.push_back(NULL);
		} else {
			std::set<int> unique;
			const size_t n = modelData.getNumberOfEntries(j);
			const int* indicators = modelData.getCompressedColumnVector(j);
			for (size_t j = 0; j < n; j++) { // Loop through non-zero entries only
				const int k = indicators[j];
				const int i = hPid[k];
				if (i < max) {
					unique.insert(i);
				}
			}
			auto indices = bsccs::make_shared<IndexVector>(unique.begin(), unique.end());
            sparseIndices.push_back(indices);
		}
	}
}

void AbstractModelSpecifics::initialize(
		int iN,
		int iK,
		int iJ,
		const CompressedDataMatrix*,
		real* iNumerPid,
		real* iNumerPid2,
		real* iDenomPid,
//		int* iNEvents,
		real* iXjY,
		std::vector<std::vector<int>* >* iSparseIndices,
		const int* iPid_unused,
		real* iOffsExpXBeta,
		real* iXBeta,
		real* iOffs,
		real* iBeta,
		const real* iY_unused//,
//		real* iWeights
		) {
	N = iN;
	K = iK;
	J = iJ;
	offsExpXBeta.resize(K);
	hXBeta.resize(K); // PT OF DIFFERENCE

	if (allocateXjY()) {
		hXjY.resize(J);
	}

	if (allocateXjX()) {
		hXjX.resize(J);
	}

	if (initializeAccumulationVectors()) {
		setPidForAccumulation(nullptr); // calls setupSparseIndices() before returning
 	} else {
		// TODO Suspect below is not necessary for non-grouped data.
		// If true, then fill with pointers to CompressedDataColumn and do not delete in destructor
		setupSparseIndices(N); // Need to be recomputed when hPid change!
	}



	size_t alignedLength = getAlignedLength(N);
// 	numerDenomPidCache.resize(3 * alignedLength, 0);
// 	numerPid = numerDenomPidCache.data();
// 	denomPid = numerPid + alignedLength; // Nested in denomPid allocation
// 	numerPid2 = numerPid + 2 * alignedLength;
	denomPid.resize(alignedLength);
	numerPid.resize(alignedLength);
	numerPid2.resize(alignedLength);

}

} // namespace
