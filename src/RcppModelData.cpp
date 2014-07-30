/*
 * RcppModelData.cpp
 *
 *  Created on: Apr 24, 2014
 *      Author: msuchard
 */

#include "Rcpp.h"
#include "RcppModelData.h"
#include "Timer.h"
#include "RcppCcdInterface.h"
#include "io/NewGenericInputReader.h"
#include "RcppProgressLogger.h"
#include "SqlModelData.h"

using namespace Rcpp;
 
XPtr<bsccs::ModelData> parseEnvironmentForPtr(const Environment& x) {
	if (!x.inherits("ccdData")) {
		::Rf_error("Input must be a ccdData object");	// TODO Change to "stop"
	}
			
	SEXP tSexp = x["ccdDataPtr"];
	if (TYPEOF(tSexp) != EXTPTRSXP) {
		::Rf_error("Input must contain a ccdDataPtr object"); // TODO Change to "stop" (bug in Rcpp 0.11)
	}	

	XPtr<bsccs::ModelData> ptr(tSexp);
	if (!ptr) {
		::Rf_error("ccdData object is uninitialized"); // TODO Change to "stop"
	}
	return ptr;	
}

XPtr<bsccs::RcppModelData> parseEnvironmentForRcppPtr(const Environment& x) {
	if (!x.inherits("ccdData")) {
		::Rf_error("Input must be a ccdData object");	// TODO Change to "stop"
	}
			
	SEXP tSexp = x["ccdDataPtr"];
	if (TYPEOF(tSexp) != EXTPTRSXP) {
		::Rf_error("Input must contain a ccdDataPtr object"); // TODO Change to "stop" (bug in Rcpp 0.11)
	}	

	XPtr<bsccs::RcppModelData> ptr(tSexp);
	if (!ptr) {
		::Rf_error("ccdData object is uninitialized"); // TODO Change to "stop"
	}
	return ptr;	
}

//' @title Print row identifiers
//' 
//' @description
//' \code{printCcdRowIds} return the row identifiers in an OHDSI CCD data object
//' 
//' @param object    An OHDSI CCD data object
//'
// [[Rcpp::export("printCcdRowIds")]]
void ccdPrintRowIds(Environment object) {
	XPtr<bsccs::RcppModelData> data = parseEnvironmentForRcppPtr(object);
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
		::Rf_error("Input must be an Rcpp externalptr"); // TODO Change to "stop"
	}
	XPtr<int> ptr(x); 
	return !ptr;
}

//' @title Get number of strata
//' 
//' @description
//' \code{getNumberOfStrata} return the number of unique strata in an OHDSI CCD data object
//' 
//' @param object    An OHDSI CCD data object
//' 
// [[Rcpp::export("getNumberOfStrata")]]
int ccdGetNumberOfStrata(Environment object) {			
	XPtr<bsccs::ModelData> data = parseEnvironmentForPtr(object);	
	return static_cast<int>(data->getNumberOfPatients());
}

//' @title Get covariate identifiers
//' 
//' @description
//' \code{getCovariateIds} returns a vector of integer covariate identifiers in an OHDSI CCD data object
//' 
//' @param object    An OHDSI CCD data object
//' 
// [[Rcpp::export("getCovariateIds")]]
std::vector<int64_t> ccdGetCovariateIds(Environment object) {
    using namespace bsccs;
	XPtr<ModelData> data = parseEnvironmentForPtr(object);
	ProfileVector covariates;
	size_t i = 0;
	if (data->getHasOffsetCovariate()) i++;
// 	if (data->getHasInterceptCovariate()) i++;
	for (; i < data->getNumberOfColumns(); ++i) {
		covariates.push_back(data->getColumn(i).getNumericalLabel());
	}
	return covariates;
}

//' @title Get covariate types
//' 
//' @description
//' \code{getCovariateTypes} returns a vector covariate types in an OHDSI CCD data object
//' 
//' @param object    An OHDSI CCD data object
//' @param covariateLabel Integer vector: covariate identifiers to return
//' 
// [[Rcpp::export("getCovariateTypes")]]
CharacterVector ccdGetCovariateType(Environment object, const std::vector<int64_t>& covariateLabel) {
    using namespace bsccs;
	XPtr<bsccs::RcppModelData> data = parseEnvironmentForRcppPtr(object);
	CharacterVector types(covariateLabel.size());	
	
	for (size_t i = 0; i < covariateLabel.size(); ++i) {
		size_t index = data->getColumnIndex(covariateLabel[i]);		
		types[i] = data->getColumn(index).getTypeString();
	}
	return types;
}

//' @title Get total number of covariates
//' 
//' @description
//' \code{getNumberOfCovariates} returns the total number of covariates in an OHDSI CCD data object
//' 
//' @param object    An OHDSI CCD data object
//'
// [[Rcpp::export("getNumberOfCovariates")]]
int ccdGetNumberOfColumns(Environment object) {	
	XPtr<bsccs::ModelData> data = parseEnvironmentForPtr(object);	
	return static_cast<int>(data->getNumberOfColumns());
}

//' @title Get total number of rows
//' 
//' @description
//' \code{getNumberOfRows} returns the total number of outcome rows in an OHDSI CCD data object
//' 
//' @param object    An OHDSI CCD data object
//'
// [[Rcpp::export("getNumberOfRows")]]
int ccdGetNumberOfRows(Environment object) {	
	XPtr<bsccs::ModelData> data = parseEnvironmentForPtr(object);	
	return static_cast<int>(data->getNumberOfRows());
}

// [[Rcpp::export(".ccdSumByGroup")]]
List ccdSumByGroup(Environment x, const std::vector<long>& covariateLabel, 
		const long groupByLabel, const int power) {
	XPtr<bsccs::RcppModelData> data = parseEnvironmentForRcppPtr(x);
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

// [[Rcpp::export(".ccdSumByStratum")]]
List ccdSumByStratum(Environment x, const std::vector<long>& covariateLabel,
		const int power) {
	XPtr<bsccs::RcppModelData> data = parseEnvironmentForRcppPtr(x);
    List list(covariateLabel.size());
    IntegerVector names(covariateLabel.size());
    for (size_t i = 0; i < covariateLabel.size(); ++i) {
        std::vector<double> result;
        data->sumByGroup(result, covariateLabel[i], power);
        list[i] = result;
        names[i] = covariateLabel[i];
    }
	list.attr("names") = names;
	return list;
}

// [[Rcpp::export(".ccdSum")]]
std::vector<double> ccdSum(Environment x, const std::vector<long>& covariateLabel,
		const int power) {
	XPtr<bsccs::RcppModelData> data = parseEnvironmentForRcppPtr(x);
	std::vector<double> result;
	for (std::vector<long>::const_iterator it = covariateLabel.begin();
	        it != covariateLabel.end(); ++it) {
	    result.push_back(data->sum(*it, power));
	}
	return result;
}

// [[Rcpp::export(".ccdNewSqlData")]]
List ccdNewSqlData(const std::string& modelTypeName, const std::string& noiseLevel) {
	using namespace bsccs;
	
	NoiseLevels noise = RcppCcdInterface::parseNoiseLevel(noiseLevel);
	bool silent = (noise == SILENT);

    ModelType modelType = RcppCcdInterface::parseModelType(modelTypeName);
	SqlModelData* ptr = new SqlModelData(modelType,
        bsccs::make_shared<loggers::RcppProgressLogger>(silent),
        bsccs::make_shared<loggers::RcppErrorHandler>());
        
	XPtr<SqlModelData> sqlModelData(ptr);

    List list = List::create(
            Rcpp::Named("ccdDataPtr") = sqlModelData
        );
    return list;
}

// [[Rcpp::export(".ccdSetHasIntercept")]]
void ccdSetHasIntercept(Environment x, bool hasIntercept) {
    using namespace bsccs;
    XPtr<ModelData> data = parseEnvironmentForPtr(x);
    data->setHasInterceptCovariate(hasIntercept);
}

// [[Rcpp::export(".ccdGetHasIntercept")]]
bool ccdGetHasIntercept(Environment x) {
    using namespace bsccs;
    XPtr<ModelData> data = parseEnvironmentForPtr(x);
    return data->getHasInterceptCovariate(); 
}

// [[Rcpp::export(".ccdFinalizeData")]]
void ccdFinalizeData(
        Environment x,
        bool addIntercept, 
        SEXP sexpOffsetCovariate,
        bool offsetAlreadyOnLogScale, 
        bool sortCovariates,
        SEXP sexpCovariatesDense,
        bool magicFlag = false) {
    using namespace bsccs;
    XPtr<ModelData> data = parseEnvironmentForPtr(x);
    
    if (data->getIsFinalized()) {
        ::Rf_error("OHDSI data object is already finalized");
    }
    
    if (addIntercept) {
        if (data->getHasInterceptCovariate()) {
            ::Rf_error("OHDSI data object already has an intercept");
        }
        // TODO add intercept as INTERCEPT_TYPE if magicFlag ==  true
        data->insert(0, DENSE); // add to front, TODO fix if offset
        data->setHasInterceptCovariate(true);
       // CompressedDataColumn& intercept = data->getColumn(0);
        const size_t numRows = data->getNumberOfRows();
        for (size_t i = 0; i < numRows; ++i) {
            data->getColumn(0).add_data(i, static_cast<real>(1.0));
        }
    }
    
    if (!Rf_isNull(sexpOffsetCovariate)) {
        // TODO handle offset
        IdType covariate = as<IdType>(sexpOffsetCovariate);
        int index = data->getColumnIndexByName(covariate);         		
 		if (index == -1) {
            std::ostringstream stream;
 			stream << "Variable " << covariate << " not found.";
            ::Rf_error(stream.str().c_str());
 			//error->throwError(stream); 
        }
        data->moveToFront(index);
        data->getColumn(0).add_label(-1); // TODO Generic label for offset?
        data->setHasOffsetCovariate(true);   
    } 
    
    if (data->getHasOffsetCovariate() && !offsetAlreadyOnLogScale) {
        ::Rf_error("Transforming the offset is not yet implemented");
    }
    
    if (!Rf_isNull(sexpCovariatesDense)) {
        // TODO handle dense conversion
        ProfileVector covariates = as<ProfileVector>(sexpCovariatesDense);
        for (auto it = covariates.begin(); it != covariates.end(); ++it) {
        	IdType index = data->getColumnIndex(*it);
        	data->getColumn(index).convertColumnToDense(data->getNumberOfRows());      
        }             
    }
                        
    data->setIsFinalized(true);
}                        

// NOTE:  IdType does not get exported into RcppExports, so hard-coded here
// TODO Could use SEXP signature and cast in function

// [[Rcpp::export(".appendSqlCcdData")]]
int ccdAppendSqlData(Environment x,
        const std::vector<int64_t>& oStratumId, 
        const std::vector<int64_t>& oRowId,
        const std::vector<double>& oY,
        const std::vector<double>& oTime,
        const std::vector<int64_t>& cRowId,
        const std::vector<int64_t>& cCovariateId,
        const std::vector<double>& cCovariateValue) {
        // o -> outcome, c -> covariates

    using namespace bsccs;
    XPtr<ModelData> data = parseEnvironmentForPtr(x);
    size_t count = data->append(oStratumId, oRowId, oY, oTime, cRowId, cCovariateId, cCovariateValue);
    return static_cast<int>(count);
}


// [[Rcpp::export(".ccdGetInterceptLabel")]]
SEXP ccdGetInterceptLabel(Environment x) {
    using namespace bsccs;
    XPtr<ModelData> data = parseEnvironmentForPtr(x);
    if (data->getHasInterceptCovariate()) {
        size_t index = data->getHasOffsetCovariate() ? 1 : 0;
        return Rcpp::wrap(data->getColumn(index).getNumericalLabel());
    } else {
        return R_NilValue;
    }
}

// [[Rcpp::export(".ccdReadData")]]
List ccdReadFileData(const std::string& fileName, const std::string& modelTypeName) {

		using namespace bsccs;
		Timer timer; 
    ModelType modelType = RcppCcdInterface::parseModelType(modelTypeName);        		
    InputReader* reader = new NewGenericInputReader(modelType,
    	bsccs::make_shared<loggers::RcppProgressLogger>(true), // make silent
    	bsccs::make_shared<loggers::RcppErrorHandler>());	
		reader->readFile(fileName.c_str()); // TODO Check for error

    XPtr<ModelData> ptr(reader->getModelData());
    
    
    const std::vector<double>& y = ptr->getYVectorRef();
    double total = std::accumulate(y.begin(), y.end(), 0.0);
    // delete reader; // TODO Test
       
    double time = timer();
    List list = List::create(
            Rcpp::Named("ccdDataPtr") = ptr,            
            Rcpp::Named("timeLoad") = time,
            Rcpp::Named("debug") = List::create(
            		Rcpp::Named("totalY") = total
            )
    );
    return list;
}

// [[Rcpp::export(".ccdModelData")]]
List ccdModelData(SEXP pid, SEXP y, SEXP z, SEXP offs, SEXP dx, SEXP sx, SEXP ix, 
    const std::string& modelTypeName,
    bool useTimeAsOffset = false) {

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

    XPtr<RcppModelData> ptr(new RcppModelData(modelType, ipid, iy, iz, ioffs, dxv, siv, spv, sxv, iiv, ipv, useTimeAsOffset));

	double duration = timer();

 	List list = List::create(
 			Rcpp::Named("data") = ptr,
 			Rcpp::Named("timeLoad") = duration
 		);
  return list;
}

namespace bsccs {

RcppModelData::RcppModelData(
        ModelType _modelType,
		const IntegerVector& _pid,
		const NumericVector& _y,
		const NumericVector& _z,
		const NumericVector& _time,
		const NumericVector& dxv, // dense
		const IntegerVector& siv, // sparse
		const IntegerVector& spv,
		const NumericVector& sxv,
		const IntegerVector& iiv, // indicator
		const IntegerVector& ipv,
		bool useTimeAsOffset
		) : ModelData(
                _modelType,
				_pid,
				_y,
				_z,
				_time,
				bsccs::make_shared<loggers::RcppProgressLogger>(),
				bsccs::make_shared<loggers::RcppErrorHandler>()
				) {
	if (useTimeAsOffset) {
	    // offset
        real_vector* r = new real_vector();
        push_back(NULL, r, DENSE);   
        r->assign(offs.begin(), offs.end()); // TODO Should not be necessary with shared_ptr
        setHasOffsetCovariate(true);
	    getColumn(0).add_label(-1);
	}			
				
	// Convert dense
	int nCovariates = static_cast<int>(dxv.size() / y.size());
	for (int i = 0; i < nCovariates; ++i) {
		push_back(
				static_cast<IntegerVector::iterator>(NULL), static_cast<IntegerVector::iterator>(NULL),
				dxv.begin() + i * y.size(), dxv.begin() + (i + 1) * y.size(),
				DENSE);
		getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));
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
        getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));				
	}

	// Convert indicator
	nCovariates = ipv.size() - 1;
	for (int i = 0; i < nCovariates; ++i) {

		int begin = ipv[i];
		int end = ipv[i + 1];

		push_back(
				iiv.begin() + begin, iiv.begin() + end,
				static_cast<NumericVector::iterator>(NULL), static_cast<NumericVector::iterator>(NULL),
				INDICATOR);
        getColumn(getNumberOfColumns() - 1).add_label(getNumberOfColumns() - (getHasOffsetCovariate() ? 1 : 0));				
	}

	this->nRows = y.size();
	
	// Clean out PIDs
	std::vector<int>& cpid = getPidVectorRef();
	
	if (cpid.size() == 0) {
	    for (size_t i = 0; i < nRows; ++i) {
	        cpid.push_back(i);
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

double RcppModelData::sum(const IdType covariate, int power) {
    size_t index = getColumnIndex(covariate); 
    if (power == 0) {  
		return reduce(index, ZeroPower());
	} else if (power == 1) {
		return reduce(index, FirstPower());
	} else {
		return reduce(index, SecondPower());	
	}
}

void RcppModelData::sumByGroup(std::vector<double>& out, const IdType covariate, const IdType groupBy, int power) {
    size_t covariateIndex = getColumnIndex(covariate);
    size_t groupByIndex = getColumnIndex(groupBy);
    out.resize(2);
    if (power == 0) {
    	reduceByGroup(out, covariateIndex, groupByIndex, ZeroPower());
    } else if (power == 1) {
  		reduceByGroup(out, covariateIndex, groupByIndex, FirstPower());  
    } else {
    	reduceByGroup(out, covariateIndex, groupByIndex, SecondPower());    
    }   
}

void RcppModelData::sumByGroup(std::vector<double>& out, const IdType covariate, int power) {
    size_t covariateIndex = getColumnIndex(covariate);
    out.resize(nPatients);
    if (power == 0) {
    	reduceByGroup(out, covariateIndex, pid, ZeroPower());
    } else if (power == 1) {
  		reduceByGroup(out, covariateIndex, pid, FirstPower());  
    } else {
    	reduceByGroup(out, covariateIndex, pid, SecondPower());    
    }    
}


RcppModelData::~RcppModelData() {
//	std::cout << "~RcppModelData() called." << std::endl;
}

} /* namespace bsccs */
