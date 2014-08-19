/*
 * AbstractModelSpecifics.cpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#include "AbstractModelSpecifics.h"
#include "ModelData.h"
#include "engine/ModelSpecifics.h"
// #include "io/InputReader.h"

//#include "Rcpp.h"

namespace bsccs {

AbstractModelSpecifics* AbstractModelSpecifics::factory(const ModelType modelType, ModelData* modelData) {
	AbstractModelSpecifics* model = nullptr;
 	switch (modelType) {
 		case ModelType::SELF_CONTROLLED_MODEL :
 			model = new ModelSpecifics<SelfControlledCaseSeries<real>,real>(*modelData);
 			break;
 		case ModelType::CONDITIONAL_LOGISTIC :
 			model = new ModelSpecifics<ConditionalLogisticRegression<real>,real>(*modelData);
 			break;
 		case ModelType::LOGISTIC :
 			model = new ModelSpecifics<LogisticRegression<real>,real>(*modelData);
 			break;
 		case ModelType::NORMAL : 
 			model = new ModelSpecifics<LeastSquares<real>,real>(*modelData);
 			break;
 		case ModelType::POISSON :
 			model = new ModelSpecifics<PoissonRegression<real>,real>(*modelData);
 			break;
		case ModelType::CONDITIONAL_POISSON :
 			model = new ModelSpecifics<ConditionalPoissonRegression<real>,real>(*modelData);
 			break; 			
 		case ModelType::COX_RAW : 		   
 			model = new ModelSpecifics<CoxProportionalHazards<real>,real>(*modelData);
 			break;
 		case ModelType::COX : 		   
 			model = new ModelSpecifics<BreslowTiedCoxProportionalHazards<real>,real>(*modelData);
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
	
	
	if (initializeAccumulationVectors()) {
        int lastPid = hPid[0];
        real lastTime = hOffs[0];
        real lastEvent = hY[0];        
        
        int pid = hPid[0] = 0;
        
	    for (int k = 1; k < K; ++k) {
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
	    }
	    pid++;
	    accReset.push_back(pid);
	    
	    // Save number of denominators
	    N = pid;
	    
	    
	    
//         std::cout << "Reset locations:";
//         std::for_each(begin(accReset), end(accReset), [](int i) {
//             std::cout << " " << i;
//         });
//         std::cout << std::endl;
	}
		
// 	if (true /* initializeTies() */) {	
// 		real lastTime = hOffs[0];
// 		real lastEvent = hY[0];
// 	
// 		std::cout << "K = " << K << std::endl;
// 		std::cout << "N = " << N << std::endl;
// 		
// //		Rcpp::stop("1");
// 						
// 		int startTie = 0;
// 		int endTie = 0;		
// 		bool inTie = false;
// 		
// 		for (int k = 1; k < K; ++k) {
// 			bool addTieToList = false;
// 			if (lastEvent == 1.0 && lastTime == hOffs[k] && lastEvent == hY[k]) {
// 				if (!inTie) {
// 					startTie = k - 1;
// 					inTie = true;
// 				} 
// 				endTie = k;		
// 			} else { // not equal
// 				if (inTie) {
// 					endTie = k - 1;
// 					inTie = false;
// 					addTieToList = true;	
// 				}
// 			}
// 			if (inTie && k == K - 1) {
// 				addTieToList = true;
// 			}			
// 			lastTime = hOffs[k];
// 			lastEvent = hY[k];
// 			if (addTieToList) {
// 				TimeTie tie{startTie,endTie};
// 				ties.push_back(tie);
// 				beginTies.push_back(startTie);
// 				endTies.push_back(endTie + 1);
// 			}
// 		}
// 		
// // 		std::cout << "Ties: " << ties.size() << std::endl;
// // 		std::for_each(begin(ties), end(ties), [](std::vector<int>& tie) {
// // 			std::cout << tie[0] << ":" << tie[1] << std::endl;
// // 		});
// // 		std::cout << std::endl;
// // 		std::for_each(begin(beginTies), end(beginTies), [](int begin) {
// // 			std::cout << " " << begin;
// // 		});
// // 		std::cout << std::endl;
// // 		std::for_each(begin(endTies), end(endTies), [](int end) {
// // 			std::cout << " " << end;
// // 		});
// // 		std::cout << std::endl;
// 	}

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
