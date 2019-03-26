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

#ifdef HAVE_OPENCL
#include "engine/GpuModelSpecifics.hpp"
#endif // HAVE_OPENCL

namespace bsccs {

template <class Model, typename RealType, class ModelG>
AbstractModelSpecifics* deviceFactory(
        const ModelData<RealType>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {
    AbstractModelSpecifics* model = nullptr;

    switch (deviceType) {
    case DeviceType::CPU :
        model = new ModelSpecifics<Model,RealType>(modelData);
        break;
#ifdef HAVE_OPENCL
    case DeviceType::GPU :
        model = new GpuModelSpecifics<Model,RealType, ModelG>(modelData, deviceName);
        break;
#endif // HAVE_OPENCL
    default:
        break; // nullptr
    }
    return model;
}

template <typename RealType>
AbstractModelSpecifics* precisionFactory(
        const ModelType modelType,
        const ModelData<RealType>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName);

template <>
AbstractModelSpecifics* precisionFactory<float>(
        const ModelType modelType,
        const ModelData<float>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {

    AbstractModelSpecifics* model = nullptr;

    switch (modelType) {
    case ModelType::SELF_CONTROLLED_MODEL : // TODO does this work?
        model =  deviceFactory<SelfControlledCaseSeries<float>,float,SelfControlledCaseSeriesG>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_LOGISTIC : // TODO does this work?
        model =  deviceFactory<ConditionalLogisticRegression<float>,float,ConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::EFRON_TIED_CONDITIONAL_LOGISTIC : // TODO does this work?
        model =  deviceFactory<EfronConditionalLogisticRegression<float>,float,EfronConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::TIED_CONDITIONAL_LOGISTIC : // TODO does this work?
        model =  deviceFactory<TiedConditionalLogisticRegression<float>,float,TiedConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::LOGISTIC :
        model = deviceFactory<LogisticRegression<float>,float,LogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::NORMAL : // TODO does this work?
        model = deviceFactory<LeastSquares<float>,float,LeastSquaresG>(modelData, deviceType, deviceName);
        break;
    case ModelType::POISSON :
        model = deviceFactory<PoissonRegression<float>,float,PoissonRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_POISSON :
        model = deviceFactory<ConditionalPoissonRegression<float>,float,ConditionalPoissonRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX_RAW : // TODO does this work?
        model = deviceFactory<CoxProportionalHazards<float>,float,CoxProportionalHazardsG>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX :
        model = deviceFactory<BreslowTiedCoxProportionalHazards<float>,float,BreslowTiedCoxProportionalHazardsG>(modelData, deviceType, deviceName);
        break;
    default:
        break;
    }

    return model;
}

template <>
AbstractModelSpecifics* precisionFactory<double>(
        const ModelType modelType,
        const ModelData<double>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {

    AbstractModelSpecifics* model = nullptr;

    switch (modelType) {
    case ModelType::SELF_CONTROLLED_MODEL :
        model =  deviceFactory<SelfControlledCaseSeries<double>,double,SelfControlledCaseSeriesG>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_LOGISTIC :
        model =  deviceFactory<ConditionalLogisticRegression<double>,double,ConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::EFRON_TIED_CONDITIONAL_LOGISTIC :
        model =  deviceFactory<EfronConditionalLogisticRegression<double>,double,EfronConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::TIED_CONDITIONAL_LOGISTIC :
        model =  deviceFactory<TiedConditionalLogisticRegression<double>,double,TiedConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::LOGISTIC :
        model = deviceFactory<LogisticRegression<double>,double,LogisticRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::NORMAL :
        model = deviceFactory<LeastSquares<double>,double,LeastSquaresG>(modelData, deviceType, deviceName);
        break;
    case ModelType::POISSON :
        model = deviceFactory<PoissonRegression<double>,double,PoissonRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_POISSON :
        model = deviceFactory<ConditionalPoissonRegression<double>,double,ConditionalPoissonRegressionG>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX_RAW :
        model = deviceFactory<CoxProportionalHazards<double>,double,CoxProportionalHazardsG>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX :
        model = deviceFactory<BreslowTiedCoxProportionalHazards<double>,double,BreslowTiedCoxProportionalHazardsG>(modelData, deviceType, deviceName);
        break;
    default:
        break;
    }
    return model;
}

AbstractModelSpecifics* AbstractModelSpecifics::factory(const ModelType modelType,
                                                        const AbstractModelData& abstractModelData,
                                                        const DeviceType deviceType,
                                                        const std::string& deviceName) {
    AbstractModelSpecifics* model = nullptr;

    if (modelType != ModelType::LOGISTIC && deviceType == DeviceType::GPU) {
        return model; // Implementing lr first on GPU.
    }

    switch(abstractModelData.getPrecisionType()) {
    case PrecisionType::FP64 :
        model = precisionFactory<double>(modelType,
                                         static_cast<const ModelData<double>&>(abstractModelData),
                                         deviceType, deviceName);
        break;
    case PrecisionType::FP32 :
        model = precisionFactory<float>(modelType,
                                        static_cast<const ModelData<float>&>(abstractModelData),
                                        deviceType, deviceName);
        break;
    default :
        break;
    }

    return model;
}

AbstractModelSpecifics::AbstractModelSpecifics(const AbstractModelData& input)
	: hPidOriginal(input.getPidVectorRef()), hPid(const_cast<int*>(hPidOriginal.data())),
      hPidSize(hPidOriginal.size()),
      boundType(MmBoundType::METHOD_2),
      algorithmType(AlgorithmType::CCD) {
    // modelData(input), hY(input.getYVectorRef()), hOffs(input.getTimeVectorRef()),
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

void AbstractModelSpecifics::setAlgorithmType(AlgorithmType alg) {
	algorithmType = alg;
}

void AbstractModelSpecifics::setLogSum(bool logSum) {
	useLogSum = logSum;
}

int AbstractModelSpecifics::getAlignedLength(int N) {
	return (N / 16) * 16 + (N % 16 == 0 ? 0 : 16);
}

// TODO take out?
template <typename RealType>
void AbstractModelSpecifics::setPidForAccumulation(const RealType* weights) {

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

// TODO take out?
template <typename RealType>
void AbstractModelSpecifics::setPidForAccumulation(const RealType* weights, int cvIndex) {

	hPidInternalPool[cvIndex] =  hPidOriginal; // Make copy
	hPidPool[cvIndex] = hPidInternalPool[cvIndex].data(); // Point to copy
	accResetPool[cvIndex].clear();

	const int ignore = -1;

	// Find first non-zero weight
	size_t index = 0;
	while(weights != nullptr && weights[index] == 0.0 && index < K) {
		hPidPool[cvIndex][index] = ignore;
		index++;
	}

	int lastPid = hPidPool[cvIndex][index];
	real lastTime = hOffs[index];
	real lastEvent = hY[index];

	int pid = hPidPool[cvIndex][index] = 0;

	for (size_t k = index + 1; k < K; ++k) {
		if (weights == nullptr || weights[k] != 0.0) {
			int nextPid = hPidPool[cvIndex][k];

			if (nextPid != lastPid) { // start new strata
				pid++;
				accResetPool[cvIndex].push_back(pid);
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

			hPidPool[cvIndex][k] = pid;
		} else {
			hPidPool[cvIndex][k] = ignore;
		}
	}
	pid++;
	accResetPool[cvIndex].push_back(pid);

	// Save number of denominators
	N = pid;

	if (weights != nullptr) {
		for (size_t i = 0; i < K; ++i) {
			if (hPidPool[cvIndex][i] == ignore) hPidPool[cvIndex][i] = N; // do NOT accumulate, since loops use: i < N
		}
	}
	setupSparseIndices(N, cvIndex); // ignore pid == N (pointing to removed data strata)
}

// TODO take out?
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
				const int i = hPid[k];  // TODO container-overflow #Generate some simulated data: #Fit the model
				if (i < max) {
					unique.insert(i);
				}
			}
			auto indices = bsccs::make_shared<IndexVector>(unique.begin(), unique.end());
            sparseIndices.push_back(indices);
		}
	}
/*
	std::cout << "setup sparse indices ";
	for (int j = 0; j < J; j++) {
		GenericIterator blah(*sparseIndices[j]);
		std::cout << "\n";
		for (auto x : blah) {
			std::cout << x << " ";
		}
	}
	std::cout << "\n";
	*/
}

// TODO take out?
void AbstractModelSpecifics::setupSparseIndices(const int max, int cvIndex) {
	sparseIndicesPool[cvIndex].clear(); // empty if full!

	for (size_t j = 0; j < J; ++j) {
		if (modelData.getFormatType(j) == DENSE || modelData.getFormatType(j) == INTERCEPT) {
			sparseIndicesPool[cvIndex].push_back(NULL);
		} else {
			std::set<int> unique;
			const size_t n = modelData.getNumberOfEntries(j);
			const int* indicators = modelData.getCompressedColumnVector(j);
			for (size_t j = 0; j < n; j++) { // Loop through non-zero entries only
				const int k = indicators[j];
				const int i = hPidPool[cvIndex][k];  // TODO container-overflow #Generate some simulated data: #Fit the model
				if (i < max) {
					unique.insert(i);
				}
			}
			auto indices = bsccs::make_shared<IndexVector>(unique.begin(), unique.end());
            sparseIndicesPool[cvIndex].push_back(indices);
		}
	}
/*
	std::cout << "setup syncCV sparse indices " << cvIndex << " ";
	for (int j = 0; j < J; j++) {
		GenericIterator blah(*sparseIndicesPool[cvIndex][j]);
	    std::cout << "\n";
		for (auto x : blah) {
			std::cout << x << " ";
		}
	}
	std::cout << "\n";
	*/

}

// TODO take out?
void AbstractModelSpecifics::deviceInitialization() {
    // Do nothing
}

// TODO take out?
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
	hXBeta.resize(K);
	//hBeta.resize(K);

	if (allocateXjY()) {
		hXjY.resize(J);
	}

	if (allocateXjX()) {
		hXjX.resize(J);
	}

	if (initializeAccumulationVectors()) {
		setPidForAccumulation(static_cast<double*>(nullptr)); // calls setupSparseIndices() before returning
 	} else {
		// TODO Suspect below is not necessary for non-grouped data.
		// If true, then fill with pointers to CompressedDataColumn and do not delete in destructor
		setupSparseIndices(N); // Need to be recomputed when hPid change!
	}



	size_t alignedLength = getAlignedLength(N + 1);
// 	numerDenomPidCache.resize(3 * alignedLength, 0);
// 	numerPid = numerDenomPidCache.data();
// 	denomPid = numerPid + alignedLength; // Nested in denomPid allocation
// 	numerPid2 = numerPid + 2 * alignedLength;
	denomPid.resize(alignedLength);
	denomPid2.resize(alignedLength);
	numerPid.resize(alignedLength);
	numerPid2.resize(alignedLength);

	deviceInitialization();

}

} // namespace
