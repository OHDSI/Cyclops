/*
 * RcppModelData.cpp
 *
 *  Created on: Apr 24, 2014
 *      Author: msuchard
 */

#include "Rcpp.h"
#include "RcppModelData.h"
#include "Timer.h"

using namespace Rcpp;


//' @title ccd_model_data
//'
//' @description
//' \code{ccd_model_data} creates a CCD model data object
//'
//' @details
//' This function is fun.  This function currently creates a deep copy of all data.
//' Another deep copy is also then made during CCD engine initialization; one of 
//' these copies should be removed.
//'
//' @param pid               Vector of patient identifiers (function assumes these are sorted)
//' @param y								 Vector of outcomes
//' @param z								 Vector of secondary outcomes (or NULL if unneeded for model)
//' @param offs							 Vector of regression model offsets (or NULL)
//' @param dx								 Dense matrix of covariates (or NULL)
//' @param sx							   Sparse matrix of covariates (or NULL)
//' @param ix								 Indicator matrix of covariates (or NULL)
//' 
//' @return
//' A list that contains a CCD model data object pointer and an operation duration
//' 
//' @examples
//' splitSql("SELECT * INTO a FROM b; USE x; DROP TABLE c;")
//'
//' @export
// [[Rcpp::export]]
List ccd_model_data(SEXP pid, SEXP y, SEXP z, SEXP offs, SEXP dx, SEXP sx, SEXP ix) {

	bsccs::Timer timer;

	IntegerVector ipid;
	if (!Rf_isNull(pid)) {
		ipid = pid; // This is not a copy
	} // else pid.size() == 0

	NumericVector iy(y);

	NumericVector iz;
	if (!Rf_isNull(z)) {
		iz = z;
	} // else z.size() == 0

	NumericVector ioffs;
	if (!Rf_isNull(offs)) {
		ioffs = offs;
	}

	// dense
	NumericVector dxv;
	if (!Rf_isNull(dx)) {
		S4 dxx(dx);
		dxv = dxx.slot("x");
	}

	// sparse
	IntegerVector siv, spv; NumericVector sxv;
	if (!Rf_isNull(sx)) {
		S4 sxx(sx);
		siv = sxx.slot("i");
		spv = sxx.slot("p");
		sxv = sxx.slot("x");
	}

	// indicator
	IntegerVector iiv, ipv;
	if (!Rf_isNull(ix)) {
		S4 ixx(ix);
		iiv = ixx.slot("i"); // TODO Check that not copy is made
		ipv = ixx.slot("p");
	}

	using namespace bsccs;
    XPtr<RcppModelData> ptr(new RcppModelData(ipid, iy, iz, ioffs, dxv, siv, spv, sxv, iiv, ipv));

	double duration = timer();

    List list = List::create(ptr, duration); // TODO Return some sort of S4 object
    return list;
}

namespace bsccs {

RcppModelData::RcppModelData(
		const IntegerVector& pid,
		const NumericVector& y,
		const NumericVector& z,
		const NumericVector& offs,
		const NumericVector& dxv, // dense
		const IntegerVector& siv, // sparse
		const IntegerVector& spv,
		const NumericVector& sxv,
		const IntegerVector& iiv, // indicator
		const IntegerVector& ipv
		) : ModelData(
				pid,
				y,
				z,
				offs
				) {
	// Convert dense
	int nCovariates = static_cast<int>(dxv.size() / y.size());
	for (int i = 0; i < nCovariates; ++i) {
		push_back(
				NULL, NULL,
				dxv.begin() + i * y.size(), dxv.begin() + (i + 1) * y.size(),
				DENSE);
		std::cout << "Added dense covariate" << std::endl;
	}

	// Convert sparse
	nCovariates = spv.size() - 1;
	for (int i = 0; i < nCovariates; ++i) {

		int begin = spv[i];
		int end = spv[i + 1];

		push_back(
				siv.begin() + begin, siv.begin() + end,
				sxv.begin() + begin, sxv.begin() + end,
				SPARSE);
		std::cout << "Added sparse covariate " << (end - begin) << std::endl;
	}

	// Convert indicator
	nCovariates = ipv.size() - 1;
	for (int i = 0; i < nCovariates; ++i) {

		int begin = ipv[i];
		int end = ipv[i + 1];

		push_back(
				iiv.begin() + begin, iiv.begin() + end,
				NULL, NULL,
				INDICATOR);
		std::cout << "Added indicator covariate " << (end - begin) << std::endl;
	}

	std::cout << "Ncol = " << getNumberOfColumns() << std::endl;

//	using bsccs::CompressedDataColumn;
//	for (int i = 0; i < getNumberOfColumns(); ++i) {
//		const CompressedDataColumn& column = getColumn(i);
//		std::cout << column.squaredSumColumn() << std::endl;
//	}

	this->nRows = y.size();

}

RcppModelData::~RcppModelData() {
	std::cout << "~RcppModelData() called." << std::endl;
}

} /* namespace bsccs */
