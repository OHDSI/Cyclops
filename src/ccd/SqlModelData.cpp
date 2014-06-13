/*
 * SqlModelData.cpp
 *
 *  Created on: Jun 12, 2014
 *      Author: msuchard
 */

// #include "Rcpp.h"
#include "SqlModelData.h"
// #include "Timer.h"
// #include "RcppCcdInterface.h"
// #include "io/NewGenericInputReader.h"
// #include "RcppProgressLogger.h"




namespace bsccs {

SqlModelData::SqlModelData(
// 		const IntegerVector& pid,
// 		const NumericVector& y,
// 		const NumericVector& z,
// 		const NumericVector& offs,
// 		const NumericVector& dxv, // dense
// 		const IntegerVector& siv, // sparse
// 		const IntegerVector& spv,
// 		const NumericVector& sxv,
// 		const IntegerVector& iiv, // indicator
// 		const IntegerVector& ipv
		) : ModelData(
// 				pid,
// 				y,
// 				z,
// 				offs
				) {
// Convert dense
// 	int nCovariates = static_cast<int>(dxv.size() / y.size());
// 	for (int i = 0; i < nCovariates; ++i) {
// 		push_back(
// 				NULL, NULL,
// 				dxv.begin() + i * y.size(), dxv.begin() + (i + 1) * y.size(),
// 				DENSE);
// 		getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns());
// //		std::cout << "Added dense covariate" << std::endl;
// 	}
// 
// Convert sparse
// 	nCovariates = spv.size() - 1;
// 	for (int i = 0; i < nCovariates; ++i) {
// 
// 		int begin = spv[i];
// 		int end = spv[i + 1];
// 
// 		push_back(
// 				siv.begin() + begin, siv.begin() + end,
// 				sxv.begin() + begin, sxv.begin() + end,
// 				SPARSE);
//         getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns());				
// //		std::cout << "Added sparse covariate " << (end - begin) << std::endl;
// 	}
// 
// Convert indicator
// 	nCovariates = ipv.size() - 1;
// 	for (int i = 0; i < nCovariates; ++i) {
// 
// 		int begin = ipv[i];
// 		int end = ipv[i + 1];
// 
// 		push_back(
// 				iiv.begin() + begin, iiv.begin() + end,
// 				NULL, NULL,
// 				INDICATOR);
//         getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns());				
// //		std::cout << "Added indicator covariate " << (end - begin) << std::endl;
// 	}
// 
// //	std::cout << "Ncol = " << getNumberOfColumns() << std::endl;
// 
// 	this->nRows = y.size();
// 	
// Clean out PIDs
// 	std::vector<int>& cpid = getPidVectorRef();
// 	
// 	if (cpid.size() == 0) {
// 	    for (int i = 0; i < nRows; ++i) {
// 	        cpid.push_back(i);
// 	    }
// 	    nPatients = nRows;
// 	} else {
//     	int currentCase = 0;
//     	int currentPID = cpid[0];
//     	cpid[0] = currentCase;
//     	for (unsigned int i = 1; i < pid.size(); ++i) {
//     	    int nextPID = cpid[i];
//     	    if (nextPID != currentPID) {
// 	            currentCase++;
//     	    }
// 	        cpid[i] = currentCase;
//     	}
//       nPatients = currentCase + 1;
//     }    
}

SqlModelData::~SqlModelData() {
//	std::cout << "~RcppModelData() called." << std::endl;
}

} /* namespace bsccs */
