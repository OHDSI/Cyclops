/*
 * RcppModelData.cpp
 *
 *  Created on: Apr 24, 2014
 *      Author: msuchard
 */

#include "Rcpp.h"
#include "RcppModelData.h"
#include "Timer.h"
#include "RcppCyclopsInterface.h"
#include "io/NewGenericInputReader.h"
#include "RcppProgressLogger.h"

using namespace Rcpp;

struct Sum {
    inline double operator()(double x, double y) {
        return x + y;
    }
};

struct ZeroPower {
    inline double operator()(double x) {
        return x == 0.0 ? 0.0 : 1.0;
    }
};

struct FirstPower {
    inline double operator()(double x) {
        return x;
    }
};

struct SecondPower {
    inline double operator()(double x) {
        return x * x;
    }
};

struct InnerProduct {
    inline double operator()(double x, double y) {
        return x * y;
    }
};

XPtr<bsccs::AbstractModelData> parseEnvironmentForPtr(const Environment& x) {
	if (!x.inherits("cyclopsData")) {
		stop("Input must be a cyclopsData object");
	}

	SEXP tSexp = x["cyclopsDataPtr"];
	if (TYPEOF(tSexp) != EXTPTRSXP) {
		stop("Input must contain a cyclopsDataPtr object");
	}

	XPtr<bsccs::AbstractModelData> ptr(tSexp);
	if (!ptr) {
		stop("cyclopsData object is uninitialized");
	}
	return ptr;
}

// XPtr<bsccs::AbstractModelData> parseEnvironmentForRcppPtr(const Environment& x) {
// 	if (!x.inherits("cyclopsData")) {
// 		stop("Input must be a cyclopsData object");
// 	}
//
// 	SEXP tSexp = x["cyclopsDataPtr"];
// 	if (TYPEOF(tSexp) != EXTPTRSXP) {
// 		stop("Input must contain a cyclopsDataPtr object");
// 	}
//
// 	XPtr<bsccs::AbstractModelData> ptr(tSexp);
// 	if (!ptr) {
// 		stop("cyclopsData object is uninitialized");
// 	}
// 	return ptr;
// }

//' @title Print row identifiers
//'
//' @description
//' \code{printCcdRowIds} return the row identifiers in a Cyclops data object
//'
//' @param object    A Cyclops data object
//'
//' @keywords internal
// [[Rcpp::export("printCyclopsRowIds")]]
void cyclopsPrintRowIds(Environment object) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
//	std::ostreamstring stream;
// 	std::vector<IdType>& rowsIds = data->get
}

// void testCcdCode(int position) {
//     std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//
//     if (position > 0 && position < static_cast<int>(v.size()))  {
//         auto reversePosition = v.size() - position - 1;
//         std::rotate(
//             v.rbegin() + reversePosition,
//             v.rbegin() + reversePosition + 1, // rotate one element
//             v.rend());
//         std::cout << "simple rotate right : ";
//         for (int n: v) std::cout << n << ' ';
//         std::cout << '\n';
//     }
// }

// [[Rcpp::export(".isRcppPtrNull")]]
bool isRcppPtrNull(SEXP x) {
	if (TYPEOF(x) != EXTPTRSXP) {
		stop("Input must be an Rcpp externalptr");
	}
	XPtr<int> ptr(x);
	return !ptr;
}

//' @title Get number of strata
//'
//' @description
//' \code{getNumberOfStrata} return the number of unique strata in a Cyclops data object
//'
//' @param object    A Cyclops data object
//'
//' @export
// [[Rcpp::export("getNumberOfStrata")]]
int cyclopsGetNumberOfStrata(Environment object) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
	return static_cast<int>(data->getNumberOfPatients());
}

//' @title Get covariate identifiers
//'
//' @description
//' \code{getCovariateIds} returns a vector of integer covariate identifiers in a Cyclops data object
//'
//' @param object    A Cyclops data object
//'
//' @export
// [[Rcpp::export("getCovariateIds")]]
std::vector<int64_t> cyclopsGetCovariateIds(Environment object) {
    using namespace bsccs;
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
	ProfileVector covariates;
	size_t i = 0;
	if (data->getHasOffsetCovariate()) i++;
// 	if (data->getHasInterceptCovariate()) i++;
	for (; i < data->getNumberOfCovariates(); ++i) {
		covariates.push_back(data->getColumnNumericalLabel(i));
	}
	return covariates;
}

//' @title Get covariate types
//'
//' @description
//' \code{getCovariateTypes} returns a vector covariate types in a Cyclops data object
//'
//' @param object    A Cyclops data object
//' @param covariateLabel Integer vector: covariate identifiers to return
//'
//' @export
// [[Rcpp::export("getCovariateTypes")]]
CharacterVector cyclopsGetCovariateType(Environment object, const std::vector<int64_t>& covariateLabel) {
    using namespace bsccs;
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
	CharacterVector types(covariateLabel.size());

	for (size_t i = 0; i < covariateLabel.size(); ++i) {
		size_t index = data->getColumnIndex(covariateLabel[i]);
		types[i] = data->getColumnTypeString(index);
	}
	return types;
}

//' @title Get floating point size
//'
//' @description
//' \code{getFloatingPointSize} returns the floating-point representation size in a Cyclops data object
//'
//' @param object   A Cyclops data object
//'
//' @export
//  [[Rcpp::export(getFloatingPointSize)]]
int cyclopsGetFloatingPointSize(Environment object) {
    XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
    return data->getFloatingPointSize();
}

//' @title Get total number of covariates
//'
//' @description
//' \code{getNumberOfCovariates} returns the total number of covariates in a Cyclops data object
//'
//' @param object    A Cyclops data object
//'
//' @export
// [[Rcpp::export("getNumberOfCovariates")]]
int cyclopsGetNumberOfColumns(Environment object) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
	auto count = data->getNumberOfCovariates();
	if (data->getHasOffsetCovariate()) {
	    --count;
	}
	return static_cast<int>(count);
}

//' @title Print Cyclops data matrix to file
//'
//' @description
//' \code{printMatrixMarket} prints the data matrix to a file
//'
//' @param object      A Cyclops data object
//' @param file        Filename
//'
//' @keywords internal
// [[Rcpp::export(printMatrixMarket)]]
void cyclopsPrintMatrixMarket(Environment object, const std::string& file) {
    XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
    std::ofstream stream(file);

    data->printMatrixMarketFormat(stream);
}

//' @title Get total number of rows
//'
//' @description
//' \code{getNumberOfRows} returns the total number of outcome rows in a Cyclops data object
//'
//' @param object    A Cyclops data object
//'
//' @export
// [[Rcpp::export("getNumberOfRows")]]
int cyclopsGetNumberOfRows(Environment object) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
	return static_cast<int>(data->getNumberOfRows());
}

//' @title Get total number of outcome types
//'
//' @description
//' \code{getNumberOfTypes} returns the total number of outcome types in a Cyclops data object
//'
//' @param object    A Cyclops data object
//'
//' @keywords internal
// [[Rcpp::export("getNumberOfTypes")]]
int cyclopsGetNumberOfTypes(Environment object) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
	return static_cast<int>(data->getNumberOfTypes());
}

// [[Rcpp::export(.cyclopsUnivariableCorrelation)]]
std::vector<double> cyclopsUnivariableCorrelation(Environment x,
                                                  const std::vector<long>& covariateLabel) {
    XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(x);
    return data->univariableCorrelation(covariateLabel);
}

// [[Rcpp::export(.cyclopsUnivariableSeparability)]]
std::vector<int> cyclopsUnivariableSeparability(Environment x,
                                                  const std::vector<long>& covariateLabel) {
    XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(x);

    std::vector<int> result;

    auto oneVariable = [&data, &result](const size_t index) {

        int value = 0;
        bsccs::FormatType formatType = data->getColumnType(index);

        if (formatType == bsccs::INTERCEPT) {

            const double xy = data->innerProductWithOutcome(index);
            const double length = static_cast<double>(data->getNumberOfRows());

            if (xy == 0.0 || xy == length) {
                value = 1;
            }

        } else if (formatType == bsccs::INDICATOR) {

            const double xy = data->innerProductWithOutcome(index);
            const double length = static_cast<double>(data->getNumberOfEntries(index));

            if (xy == 0.0 || xy == length) {
                value = 1;
            }

        } else {
            // TODO -- not yet implemented
        }

        result.push_back(value);
    };

    if (covariateLabel.size() == 0) {
        result.reserve(data->getNumberOfCovariates());
        size_t index = (data->getHasOffsetCovariate()) ? 1 : 0;
        for (; index <  data->getNumberOfCovariates(); ++index) {
            oneVariable(index);
        }
    } else {
        result.reserve(covariateLabel.size());
        for(auto it = covariateLabel.begin(); it != covariateLabel.end(); ++it) {
            oneVariable(data->getColumnIndex(*it));
        }
    }

    return result;
}

// [[Rcpp::export(".cyclopsSumByGroup")]]
List cyclopsSumByGroup(Environment x, const std::vector<long>& covariateLabel,
		const long groupByLabel, const int power) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(x);
    List list(covariateLabel.size());
    IntegerVector names(covariateLabel.size());
    for (size_t i = 0; i < covariateLabel.size(); ++i) {
        std::vector<double> result;
        data->sumByGroup(result, covariateLabel[i], groupByLabel, power);
        list[i] = result;
        names[i] = covariateLabel[i];
    }
	list.attr("names") = names;
	return list;
}

// [[Rcpp::export(".cyclopsSumByStratum")]]
List cyclopsSumByStratum(Environment x, const std::vector<long>& covariateLabel,
		const int power) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(x);
    List list(covariateLabel.size());
    IntegerVector names(covariateLabel.size());
    for (size_t i = 0; i < covariateLabel.size(); ++i) {
        std::vector<double> result;
        data->sumByPid(result, covariateLabel[i], power);
        list[i] = result;
        names[i] = covariateLabel[i];
    }
	list.attr("names") = names;
	return list;
}

// [[Rcpp::export(".cyclopsSum")]]
std::vector<double> cyclopsSum(Environment x, const std::vector<long>& covariateLabel,
		const int power) {
	XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(x);
	std::vector<double> result;
	for (std::vector<long>::const_iterator it = covariateLabel.begin();
	        it != covariateLabel.end(); ++it) {
	    result.push_back(data->sum(*it, power));
	}
	return result;
}



bsccs::AbstractModelData* factory(const bsccs::ModelType modelType, const bool silent,
                           const int floatingPoint) {
    using namespace bsccs;

    if (floatingPoint == 32) {
        return new RcppModelData<float>(modelType,
            bsccs::make_shared<loggers::RcppProgressLogger>(silent),
            bsccs::make_shared<loggers::RcppErrorHandler>());
    } else {
        return new RcppModelData<double>(modelType,
            bsccs::make_shared<loggers::RcppProgressLogger>(silent),
            bsccs::make_shared<loggers::RcppErrorHandler>());
    }
}

// [[Rcpp::export(".cyclopsNewSqlData")]]
List cyclopsNewSqlData(const std::string& modelTypeName, const std::string& noiseLevel,
                       int floatingPoint) {
	using namespace bsccs;

	NoiseLevels noise = RcppCcdInterface::parseNoiseLevel(noiseLevel);
	bool silent = (noise == SILENT);

    ModelType modelType = RcppCcdInterface::parseModelType(modelTypeName);
    AbstractModelData* ptr = factory(modelType, silent, floatingPoint);

	XPtr<AbstractModelData> sqlModelData(ptr);

    List list = List::create(
            Rcpp::Named("cyclopsDataPtr") = sqlModelData
        );
    return list;
}

// [[Rcpp::export(".cyclopsMedian")]]
double cyclopsMedian(const NumericVector& vector) {
    // Make copy
    std::vector<double> data(vector.begin(), vector.end());
    return bsccs::median(data.begin(), data.end());
}

// [[Rcpp::export(".cyclopsQuantile")]]
double cyclopsQuantile(const NumericVector& vector, double q) {
    if (q < 0.0 || q > 1.0) Rcpp::stop("Invalid quantile");
    // Make copy
    std::vector<double> data(vector.begin(), vector.end());
    return bsccs::quantile(data.begin(), data.end(), q);
}

// [[Rcpp::export(".cyclopsNormalizeCovariates")]]
std::vector<double> cyclopsNormalizeCovariates(Environment x, const std::string& normalizationName) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    NormalizationType type = RcppCcdInterface::parseNormalizationType(normalizationName);
    return data->normalizeCovariates(type);
}

// [[Rcpp::export(".cyclopsSetHasIntercept")]]
void cyclopsSetHasIntercept(Environment x, bool hasIntercept) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    data->setHasInterceptCovariate(hasIntercept);
}

// [[Rcpp::export(".cyclopsGetHasIntercept")]]
bool cyclopsGetHasIntercept(Environment x) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    return data->getHasInterceptCovariate();
}

// [[Rcpp::export(".cyclopsGetHasOffset")]]
bool cyclopsGetHasOffset(Environment x) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    return data->getHasOffsetCovariate();
}

// [[Rcpp::export(".cyclopsGetMeanOffset")]]
double cyclopsGetMeanOffset(Environment x) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    return (data->getHasOffsetCovariate()) ?
        data->sum(-1, 1) / data->getNumberOfRows() :
        0.0;
}

// [[Rcpp::export("getYVector")]]
std::vector<double> cyclopsGetYVector(Environment object) {
    XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
    return data->copyYVector();
    //XPtr<bsccs::ModelData> data = parseEnvironmentForPtr(object);
    //return (data->getYVectorRef());
}

// [[Rcpp::export("getTimeVector")]]
std::vector<double> cyclopsGetTimeVector(Environment object) {
    XPtr<bsccs::AbstractModelData> data = parseEnvironmentForPtr(object);
    return data->copyTimeVector();
    // XPtr<bsccs::ModelData> data = parseEnvironmentForPtr(object);
    // return (data->getTimeVectorRef());
}

// [[Rcpp::export(".cyclopsFinalizeData")]]
void cyclopsFinalizeData(
        Environment x,
        bool addIntercept,
        SEXP sexpOffsetCovariate,
        bool offsetAlreadyOnLogScale,
        bool sortCovariates,
        SEXP sexpCovariatesDense,
        bool magicFlag = false) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);

    if (data->getIsFinalized()) {
        ::Rf_error("OHDSI data object is already finalized");
    }

    if (addIntercept) {
        if (data->getHasInterceptCovariate()) {
            ::Rf_error("OHDSI data object already has an intercept");
        }
        // TODO add intercept as INTERCEPT_TYPE if magicFlag ==  true
        data->addIntercept();
       //  data->insert(0, DENSE); // add to front, TODO fix if offset
       //  data->setHasInterceptCovariate(true);
       // // CompressedDataColumn& intercept = data->getColumn(0);
       //  const size_t numRows = data->getNumberOfRows();
       //  for (size_t i = 0; i < numRows; ++i) {
       //      data->getColumn(0).add_data(i, static_cast<real>(1.0));
       //  }
    }

    if (!Rf_isNull(sexpOffsetCovariate)) {
        // TODO handle offset
        IdType covariate = as<IdType>(sexpOffsetCovariate);
        if (covariate != -1) {
            int index = data->getColumnIndexByName(covariate);
            if (index == -1) {
                std::ostringstream stream;
                stream << "Variable " << covariate << " not found.";
                stop(stream.str().c_str());
                //error->throwError(stream);
            }
        }
        data->setOffsetCovariate(covariate);

 //        int index;
 //        if (covariate == -1) { // TODO  Bad, magic number
 //             data->moveTimeToCovariate(true);
 //            index = data->getNumberOfCovariates() - 1;
 //        } else {
 //            index = data->getColumnIndexByName(covariate);
 // 	    	if (index == -1) {
 //                std::ostringstream stream;
 //     			stream << "Variable " << covariate << " not found.";
 //                stop(stream.str().c_str());
 //     			//error->throwError(stream);
 //            }
 //        }
 //        data->moveToFront(index);
 //        data->getColumn(0).add_label(-1); // TODO Generic label for offset?
 //        data->setHasOffsetCovariate(true);
    }

    if (data->getHasOffsetCovariate() && !offsetAlreadyOnLogScale) {
        //stop("Transforming the offset is not yet implemented");
        data->logTransformCovariate(0);
        // data->getColumn(0).transform([](real x) {
        //     return std::log(x);
        // });
    }

    if (!Rf_isNull(sexpCovariatesDense)) {
        // TODO handle dense conversion
        ProfileVector covariates = as<ProfileVector>(sexpCovariatesDense);
        for (auto it = covariates.begin(); it != covariates.end(); ++it) {
            data->convertCovariateToDense(*it);
        	// IdType index = data->getColumnIndex(*it);
        	// data->getColumn(index).convertColumnToDense(data->getNumberOfRows());
        }
    }

    data->setIsFinalized(true);
}


// [[Rcpp::export(".loadCyclopsDataY")]]
void cyclopsLoadDataY(Environment x,
        const std::vector<int64_t>& stratumId,
        const std::vector<int64_t>& rowId,
        const std::vector<double>& y,
        const std::vector<double>& time) {

    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    data->loadY(stratumId, rowId, y, time);
}

// [[Rcpp::export(".loadCyclopsDataMultipleX")]]
int cyclopsLoadDataMultipleX(Environment x,
		const std::vector<int64_t>& covariateId,
		const std::vector<int64_t>& rowId,
		const std::vector<double>& covariateValue,
		const bool checkCovariateIds,
		const bool checkCovariateBounds,
		const bool append,
		const bool forceSparse) {

	using namespace bsccs;
	XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);

	return data->loadMultipleX(covariateId, rowId, covariateValue, checkCovariateIds,
                            checkCovariateBounds, append, forceSparse);
}

// [[Rcpp::export(".loadCyclopsDataX")]]
int cyclopsLoadDataX(Environment x,
        const int64_t covariateId,
        const std::vector<int64_t>& rowId,
        const std::vector<double>& covariateValue,
        const bool replace,
        const bool append,
        const bool forceSparse) {

    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);

    // rowId.size() == 0 -> dense
    // covariateValue.size() == 0 -> indicator

    return data->loadX(covariateId, rowId, covariateValue, replace, append, forceSparse);
}

// NOTE:  IdType does not get exported into RcppExports, so hard-coded here
// TODO Could use SEXP signature and cast in function

// [[Rcpp::export(".appendSqlCyclopsData")]]
int cyclopsAppendSqlData(Environment x,
        const std::vector<int64_t>& oStratumId,
        const std::vector<int64_t>& oRowId,
        const std::vector<double>& oY,
        const std::vector<double>& oTime,
        const std::vector<int64_t>& cRowId,
        const std::vector<int64_t>& cCovariateId,
        const std::vector<double>& cCovariateValue) {
        // o -> outcome, c -> covariates

    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    size_t count = data->append(oStratumId, oRowId, oY, oTime, cRowId, cCovariateId, cCovariateValue);
    return static_cast<int>(count);
}


// [[Rcpp::export(".cyclopsGetInterceptLabel")]]
SEXP cyclopsGetInterceptLabel(Environment x) {
    using namespace bsccs;
    XPtr<AbstractModelData> data = parseEnvironmentForPtr(x);
    if (data->getHasInterceptCovariate()) {
        size_t index = data->getHasOffsetCovariate() ? 1 : 0;
        return Rcpp::wrap(data->getColumnNumericalLabel(index));
    } else {
        return R_NilValue;
    }
}

template class bsccs::ModelData<double>;
template class bsccs::ModelData<float>;


// [[Rcpp::export(".cyclopsReadData")]]
List cyclopsReadFileData(const std::string& fileName, const std::string& modelTypeName) {
		using namespace bsccs;
		Timer timer;
    ModelType modelType = RcppCcdInterface::parseModelType(modelTypeName);
    InputReader* reader = new NewGenericInputReader(modelType,
    	bsccs::make_shared<loggers::RcppProgressLogger>(true), // make silent
    	bsccs::make_shared<loggers::RcppErrorHandler>());
		reader->readFile(fileName.c_str()); // TODO Check for error

    XPtr<AbstractModelData> ptr(reader->getModelData());


    //const std::vector<double>& y = ptr->getYVectorRef();
    double total = 0.0; //std::accumulate(y.begin(), y.end(), 0.0);
    // delete reader; // TODO Test

    double time = timer();
    List list = List::create(
            Rcpp::Named("cyclopsDataPtr") = ptr,
            Rcpp::Named("timeLoad") = time,
            Rcpp::Named("debug") = List::create(
            		Rcpp::Named("totalY") = total
            )
    );
    return list;
}

// [[Rcpp::export(".cyclopsModelData")]]
List cyclopsModelData(SEXP pid, SEXP y, SEXP z, SEXP offs, SEXP dx, SEXP sx, SEXP ix,
    const std::string& modelTypeName,
    bool useTimeAsOffset = false,
    int numTypes = 1,
    int floatingPoint = 64) {

    using namespace bsccs;
    ModelType modelType = RcppCcdInterface::parseModelType(modelTypeName);

	Timer timer;

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
		iiv = ixx.slot("i"); // TODO Check that no copy is made
		ipv = ixx.slot("p");
	}

	AbstractModelData* modelData =
	    (floatingPoint == 32) ?
	    static_cast<AbstractModelData*>(new RcppModelData<float>(modelType, ipid, iy, iz, ioffs, dxv, siv,
                               spv, sxv, iiv, ipv, useTimeAsOffset, numTypes)) :
        static_cast<AbstractModelData*>(new RcppModelData<double>(modelType, ipid, iy, iz, ioffs, dxv, siv,
            spv, sxv, iiv, ipv, useTimeAsOffset, numTypes));

    XPtr<AbstractModelData> ptr(modelData);

	double duration = timer();

 	List list = List::create(
 			Rcpp::Named("data") = ptr,
 			Rcpp::Named("timeLoad") = duration
 		);
  return list;
}

namespace bsccs {

template <typename RealType>
RcppModelData<RealType>::RcppModelData(
			ModelType modelType,
	        loggers::ProgressLoggerPtr log,
    	    loggers::ErrorHandlerPtr error
) : ModelData<RealType>(modelType, log, error) { }

template <typename RealType>
RcppModelData<RealType>::RcppModelData(
        ModelType _modelType,
		const IntegerVector& _pid,
		const NumericVector& _y,
		const NumericVector& _type,
		const NumericVector& _time,
		const NumericVector& dxv, // dense
		const IntegerVector& siv, // sparse
		const IntegerVector& spv,
		const NumericVector& sxv,
		const IntegerVector& iiv, // indicator
		const IntegerVector& ipv,
		bool useTimeAsOffset,
		int numTypes
) : ModelData<RealType>(
                _modelType,
				_pid,
				_y,
				_type,
				_time,
				bsccs::make_shared<loggers::RcppProgressLogger>(),
				bsccs::make_shared<loggers::RcppErrorHandler>()
				) {
	if (useTimeAsOffset) {
	    // offset
        RealVectorPtr r = make_shared<RealVector>();
        X.push_back(NULL, r, DENSE);
        r->assign(offs.begin(), offs.end()); // TODO Should not be necessary with shared_ptr
        setHasOffsetCovariate(true);
	    X.getColumn(0).add_label(-1);
	}

    nTypes = numTypes; // TODO move into constructor

	// Convert dense
	int nCovariates = static_cast<int>(dxv.size() / y.size());
	for (int i = 0; i < nCovariates; ++i) {
		if (numTypes == 1) {
			X.push_back(
					static_cast<IntegerVector::iterator>(NULL), static_cast<IntegerVector::iterator>(NULL),
					dxv.begin() + i * y.size(), dxv.begin() + (i + 1) * y.size(),
					DENSE);
			X.getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
		} else {
			std::vector<RealVectorPtr> covariates;
			for (int c = 0; c < numTypes; ++c) {
				covariates.push_back(make_shared<RealVector>(y.size(), 0));
			}
			size_t offset = i * y.size();
			for (size_t k = 0; k < y.size(); ++k) {
				covariates[static_cast<int>(_type[k])]->at(k) = dxv[offset + k];
			}
			for (int c = 0; c < numTypes; ++c) {
				X.push_back(
// 						static_cast<IntegerVector::iterator>(NULL),static_cast<IntegerVector::iterator>(NULL),
//						covariates[c].begin(), covariates[c].end(),
                        NULL,
                        covariates[c],
						DENSE);
				X.getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
			}
		}
	}

	// Convert sparse
	nCovariates = spv.size() - 1;
	for (int i = 0; i < nCovariates; ++i) {

		int begin = spv[i];
		int end = spv[i + 1];

		if (numTypes == 1) {
		    X.push_back(
			    	siv.begin() + begin, siv.begin() + end,
				    sxv.begin() + begin, sxv.begin() + end,
    				SPARSE);
            X.getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
        } else {
			std::vector<IntVectorPtr> covariatesI;
			std::vector<RealVectorPtr> covariatesX;
			for (int c = 0; c < numTypes; ++c) {
				covariatesI.push_back(make_shared<IntVector>());
				covariatesX.push_back(make_shared<RealVector>());
				X.push_back(covariatesI[c], covariatesX[c], SPARSE);
				X.getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
			}

            auto itI = siv.begin() + begin;
            auto itX = sxv.begin() + begin;
            for (; itI != siv.begin() + end; ++itI, ++itX) {
                int type = _type[*itI];
                covariatesI[type]->push_back(*itI);
                covariatesX[type]->push_back(*itX);
            }
        }
	}

	// Convert indicator
	nCovariates = ipv.size() - 1;
	for (int i = 0; i < nCovariates; ++i) {

		int begin = ipv[i];
		int end = ipv[i + 1];

        if (numTypes == 1) {
    		X.push_back(
	    			iiv.begin() + begin, iiv.begin() + end,
		    		static_cast<NumericVector::iterator>(NULL), static_cast<NumericVector::iterator>(NULL),
			    	INDICATOR);
            X.getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
        } else {
			std::vector<IntVectorPtr> covariates;
			for (int c = 0; c < numTypes; ++c) {
				covariates.push_back(make_shared<IntVector>());
				X.push_back(covariates[c], NULL, INDICATOR);
				X.getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
			}

            for (auto it = iiv.begin() + begin; it != iiv.begin() + end; ++it) {
                int type = _type[*it];
                covariates[type]->push_back(*it);
            }
        }
	}

	this->getX().nRows = y.size();

	// Clean out PIDs
	std::vector<int>& cpid = getPidVectorRef();

	if (cpid.size() == 0) {
	    const auto nRows = getNumberOfRows();
	    for (size_t i = 0; i < nRows; ++i) {
	        cpid.push_back(i); // TODO These are not necessary; remove.
	    }
	    nPatients = nRows;
	} else {
    	int currentCase = 0;
    	int currentPID = cpid[0];
    	cpid[0] = currentCase;
    	for (size_t i = 1; i < pid.size(); ++i) {
    	    int nextPID = cpid[i];
    	    if (nextPID != currentPID) {
	            currentCase++;
	            currentPID = nextPID;
    	    }
	        cpid[i] = currentCase;
    	}
        nPatients = currentCase + 1;
    }
}

// template <typename RealType>
// double RcppModelData<RealType>::sum(const IdType covariate, int power) {
//
//     size_t index = getColumnIndex(covariate);
//     if (power == 0) {
// 		return reduce(index, ZeroPower());
// 	} else if (power == 1) {
// 		return reduce(index, FirstPower());
// 	} else {
// 		return reduce(index, SecondPower());
// 	}
// }

// template <typename RealType>
// void RcppModelData<RealType>::sumByGroup(std::vector<double>& out, const IdType covariate, const IdType groupBy, int power) {
//     size_t covariateIndex = getColumnIndex(covariate);
//     size_t groupByIndex = getColumnIndex(groupBy);
//     out.resize(2);
//     if (power == 0) {
//     	reduceByGroup(out, covariateIndex, groupByIndex, ZeroPower());
//     } else if (power == 1) {
//   		reduceByGroup(out, covariateIndex, groupByIndex, FirstPower());
//     } else {
//     	reduceByGroup(out, covariateIndex, groupByIndex, SecondPower());
//     }
// }

// template <typename RealType>
// void RcppModelData<RealType>::sumByGroup(std::vector<double>& out, const IdType covariate, int power) {
//     size_t covariateIndex = getColumnIndex(covariate);
//     out.resize(nPatients);
//     if (power == 0) {
//     	reduceByGroup(out, covariateIndex, pid, ZeroPower());
//     } else if (power == 1) {
//   		reduceByGroup(out, covariateIndex, pid, FirstPower());
//     } else {
//     	reduceByGroup(out, covariateIndex, pid, SecondPower());
//     }
// }

template <typename RealType>
RcppModelData<RealType>::~RcppModelData() {
//	std::cout << "~RcppModelData() called." << std::endl;
}

} /* namespace bsccs */
