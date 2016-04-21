/*
 * CyclicCoordinateDescent.cpp
 *
 *  Created on: May-June, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <time.h>
#include <set>
#include <thread>

// #include <boost/iterator/permutation_iterator.hpp>
// #include <boost/iterator/transform_iterator.hpp>
// #include <boost/iterator/zip_iterator.hpp>
//#include <boost/iterator/counting_iterator.hpp>

#include "CyclicCoordinateDescent.h"
// #include "io/InputReader.h"
#include "Iterators.h"
#include "engine/ParallelLoops.h"
// #include "io/ProgressLogger.h"

#include <sys/time.h>
#include "tbb/parallel_for.h"

//#ifdef MY_RCPP_FLAG
//	#include <R.h>
//#else
//	void Rprintf(...) {
//		// Do nothing
//	}
//#endif

#ifndef MY_RCPP_FLAG
#define PI	3.14159265358979323851280895940618620443274267017841339111328125
#else
//#include "Rcpp.h"
#endif

namespace bsccs {

using namespace std;

// void compareIntVector(int* vec0, int* vec1, int dim, const char* name) {
// 	for (int i = 0; i < dim; i++) {
// 		if (vec0[i] != vec1[i]) {
// 			cerr << "Error at " << name << "[" << i << "]: ";
// 			cerr << vec0[i] << " != " << vec1[i] << endl;
// 			exit(0);
// 		}
// 	}
// }


std::string getEnvironment( const string & var ) {
     const char * val = std::getenv( var.c_str() );
     return val == 0 ? "" : val;
}


CyclicCoordinateDescent::CyclicCoordinateDescent(
			ModelData* reader,
			AbstractModelSpecifics& specifics,
			priors::JointPriorPtr prior,
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error
		) : modelSpecifics(specifics), jointPrior(prior), logger(_logger), error(_error) {
	N = reader->getNumberOfPatients();
	K = reader->getNumberOfRows();
	J = reader->getNumberOfColumns();
	
	hXI = reader;

	hY = reader->getYVector(); // TODO Delegate all data to ModelSpecifics
//	hOffs = reader->getOffsetVector();
	hPid = reader->getPidVector();

	conditionId = reader->getConditionId();

	updateCount = 0;
	likelihoodCount = 0;
	noiseLevel = NOISY;

	init(reader->getHasOffsetCovariate());
}

CyclicCoordinateDescent::~CyclicCoordinateDescent(void) {

//	free(hPid);
//	free(hNEvents);
//	free(hY);
//	free(hOffs);
	
//	free(hBeta);
//	free(hXBeta);
//	free(hXBetaSave);
//	free(hDelta);
	
// #ifdef TEST_ROW_INDEX
// 	for (int j = 0; j < J; ++j) {
// 		if (hXColumnRowIndicators[j]) {
// 			free(hXColumnRowIndicators[j]);
// 		}
// 	}
// 	free(hXColumnRowIndicators);
// #endif

//	free(hXjY);
//	free(offsExpXBeta);
//	free(xOffsExpXBeta);
//	free(denomPid);  // Nested in numerPid allocation
//	free(numerPid);
//	free(t1);
	
// #ifdef NO_FUSE
// 	free(wPid);
// #endif
	
// 	if (hWeights) {
// 		free(hWeights);
// 	}

#ifdef SPARSE_PRODUCT
// 	for (std::vector<std::vector<int>* >::iterator it = sparseIndices.begin();
// 			it != sparseIndices.end(); ++it) {
// 		if (*it) {
// 			delete *it;
// 		}
// 	}
#endif
}

void CyclicCoordinateDescent::setNoiseLevel(NoiseLevels noise) {
	noiseLevel = noise;
}

string CyclicCoordinateDescent::getPriorInfo() {
	return jointPrior->getDescription();
}

void CyclicCoordinateDescent::setPrior(priors::JointPriorPtr newPrior) {
    jointPrior = newPrior;
}

void CyclicCoordinateDescent::resetBounds() {
	for (int j = 0; j < J; j++) {
		hDelta[j] = 2.0;
	}
}

void CyclicCoordinateDescent::init(bool offset) {
	
	// Set parameters and statistics space
	hDelta.resize(J, static_cast<double>(2.0));
	hBeta.resize(J, static_cast<double>(0.0));
	hBetaMM.resize(J, static_cast<double>(0.0));

// 	hXBeta = (real*) calloc(K, sizeof(real));
// 	hXBetaSave = (real*) calloc(K, sizeof(real));
	hXBeta.resize(K, static_cast<real>(0.0));
	hXBetaSave.resize(K, static_cast<real>(0.0));
	
	fixBeta.resize(J, false);
	
	// Recode patient ids  TODO Delegate to grouped model
	int currentNewId = 0;
	int currentOldId = hPid[0];
	
	for(int i = 0; i < K; i++) {
		if (hPid[i] != currentOldId) {			
			currentOldId = hPid[i];
			currentNewId++;
		}
		hPid[i] = currentNewId;
	}
		
	// Init temporary variables
//	offsExpXBeta = (real*) malloc(sizeof(real) * K);
//	xOffsExpXBeta = (real*) malloc(sizeof(real) * K);

	// Put numer, numer2 and denom in single memory block, with first entries on 16-word boundary
// 	int alignedLength = getAlignedLength(N);
// 	numerPid = (real*) malloc(sizeof(real) * 3 * alignedLength);
// 	denomPid = numerPid + alignedLength; // Nested in denomPid allocation
// 	numerPid2 = numerPid + 2 * alignedLength;

//	hNEvents = (int*) malloc(sizeof(int) * N);
//	hXjY = (real*) malloc(sizeof(real) * J);
	hWeights.resize(0);
	
// #ifdef NO_FUSE
// 	wPid = (real*) malloc(sizeof(real) * alignedLength);
// #endif

	// TODO Suspect below is not necessary for non-grouped data.
	// If true, then fill with pointers to CompressedDataColumn and do not delete in destructor
// 	for (int j = 0; j < J; ++j) {
// 		if (hXI->getFormatType(j) == DENSE) {
// 			sparseIndices.push_back(NULL);
// 		} else {
// 			std::set<int> unique;
// 			const int n = hXI->getNumberOfEntries(j);
// 			const int* indicators = hXI->getCompressedColumnVector(j);
// 			for (int j = 0; j < n; j++) { // Loop through non-zero entries only
// 				const int k = indicators[j];
// 				const int i = hPid[k]; // ERROR HERE IN STRAT-COX, ALSO CONSIDER TIES, MOVE TO ENGINE???
// 				unique.insert(i);
// 			}
// 			std::vector<int>* indices = new std::vector<int>(unique.begin(),
// 					unique.end());
// 			sparseIndices.push_back(indices);
// 		}
// 	}

	useCrossValidation = false;
	validWeights = false;
	sufficientStatisticsKnown = false;
	fisherInformationKnown = false;
	varianceKnown = false;
	if (offset) {
		hBeta[0] = static_cast<double>(1);
		fixBeta[0] = true;
		xBetaKnown = false;
	} else {
		xBetaKnown = true; // all beta = 0 => xBeta = 0
	}
	doLogisticRegression = false;
        
	modelSpecifics.initialize(N, K, J, hXI, NULL, NULL, NULL,
			NULL, NULL,
			hPid, NULL,
			hXBeta.data(), NULL,
			NULL,
			hY
			);
}

int CyclicCoordinateDescent::getAlignedLength(int N) {
	return (N / 16) * 16 + (N % 16 == 0 ? 0 : 16);
}

void CyclicCoordinateDescent::computeNEvents() {  
	modelSpecifics.setWeights(hWeights.data(), useCrossValidation);
}

void CyclicCoordinateDescent::resetBeta(void) {
	for (int j = 0; j < J; j++) {
		hBeta[j] = 0.0;
	}
	for (int k = 0; k < K; k++) {
		hXBeta[k] = 0.0;
	}
	sufficientStatisticsKnown = false;
}

void CyclicCoordinateDescent::logResults(const char* fileName, bool withASE) {

	ofstream outLog(fileName);
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << fileName;
		error->throwError(stream);		
	}
	string sep(","); // TODO Make option

	outLog << "label"
			<< sep << "estimate"
			//<< sep << "score"
			;
	if (withASE) {
		outLog << sep << "ASE";
	}
	outLog << endl;

	for (int i = 0; i < J; i++) {		
		outLog << hXI->getColumn(i).getLabel()
//				<< sep << conditionId
				<< sep << hBeta[i];
		if (withASE) {
			double ASE = sqrt(getAsymptoticVariance(i,i));
			outLog << sep << ASE;
		}
		outLog << endl;
	}
	outLog.flush();
	outLog.close();
}

double CyclicCoordinateDescent::getPredictiveLogLikelihood(real* weights) {

	if (!xBetaKnown) {
		computeXBeta();
		xBetaKnown = true;
		sufficientStatisticsKnown = false;
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Remove index????
		sufficientStatisticsKnown = true;
	}

	getDenominators();

	return modelSpecifics.getPredictiveLogLikelihood(weights); // TODO Pass double
}

void CyclicCoordinateDescent::getPredictiveEstimates(real* y, real* weights) const {
	modelSpecifics.getPredictiveEstimates(y, weights);
}

int CyclicCoordinateDescent::getBetaSize(void) {
	return J;
}

int CyclicCoordinateDescent::getPredictionSize(void) const {
	return K;
}

bool CyclicCoordinateDescent::getIsRegularized(int i) const {
    return jointPrior->getIsRegularized(i);    
}

double CyclicCoordinateDescent::getBeta(int i) {
	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, i);
	}
	return static_cast<double>(hBeta[i]);
}

bool CyclicCoordinateDescent::getFixedBeta(int i) {
	return fixBeta[i];
}

void CyclicCoordinateDescent::setFixedBeta(int i, bool value) {
	fixBeta[i] = value;
}

void CyclicCoordinateDescent::checkAllLazyFlags(void) {
	if (!xBetaKnown) {
		computeXBeta();
		xBetaKnown = true;
		sufficientStatisticsKnown = false;
	}

	if (!validWeights) {
		computeNEvents();
		computeFixedTermsInLogLikelihood();
		computeFixedTermsInGradientAndHessian();
		validWeights = true;
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Check index?
		sufficientStatisticsKnown = true;
	}
}

double CyclicCoordinateDescent::getLogLikelihood(void) {

	checkAllLazyFlags();

	getDenominators();
	likelihoodCount += 1;

	return modelSpecifics.getLogLikelihood(useCrossValidation);
}

void CyclicCoordinateDescent::getDenominators() {
	// Do nothing
}

double convertVarianceToHyperparameter(double value) {
	return sqrt(2.0 / value);
}

double convertHyperparameterToVariance(double value) {
	return 2.0 / (value * value);
}

void CyclicCoordinateDescent::setHyperprior(double value) {
	jointPrior->setVariance(value);
}

//Hierarchical Support
void CyclicCoordinateDescent::setClassHyperprior(double value) {
	jointPrior->setVariance(1,value);
}

double CyclicCoordinateDescent::getHyperprior(void) const {
	return jointPrior->getVariance();
}

void CyclicCoordinateDescent::makeDirty(void) {
	xBetaKnown = false;
	validWeights = false;
	sufficientStatisticsKnown = false;
}

void CyclicCoordinateDescent::setPriorType(int iPriorType) {
	if (iPriorType < priors::NONE || iPriorType > priors::NORMAL) {
	    std::ostringstream stream;
		stream << "Unknown prior type";
		error->throwError(stream);		
	}
	priorType = iPriorType;
}

void CyclicCoordinateDescent::setBeta(const std::vector<double>& beta) {
	for (int j = 0; j < J; ++j) {
		hBeta[j] = beta[j]; // TODO Use std::copy
	}
	xBetaKnown = false;
	sufficientStatisticsKnown = false;
	fisherInformationKnown = false;
	varianceKnown = false;
}

void CyclicCoordinateDescent::setBeta(int i, double beta) {
#define PROCESS_IN_MS
#ifdef PROCESS_IN_MS
	double delta = beta - hBeta[i];
	updateXBeta(delta, i);
#else // Delay and then call computeSufficientStatistics
	SetBetaEntry entry(i, hBeta[i]);
	setBetaList.push_back(entry); // save old value and index
	hBeta[i] = beta;
	xBetaKnown = false;
#endif
	fisherInformationKnown = false;
	varianceKnown = false;
}

void CyclicCoordinateDescent::setWeights(real* iWeights) {

	if (iWeights == NULL) {
		if (hWeights.size() != 0) {			
			hWeights.resize(0);
		}
		
		// Turn off weights
		useCrossValidation = false;
		validWeights = false;
		sufficientStatisticsKnown = false;
	} else {

		if (hWeights.size() == 0) {
			hWeights.resize(K); // = (real*) malloc(sizeof(real) * K);
		}
		for (int i = 0; i < K; ++i) {
			hWeights[i] = iWeights[i];
		}
		useCrossValidation = true;
		validWeights = false;
		sufficientStatisticsKnown = false;
	}
}
	
double CyclicCoordinateDescent::getLogPrior(void) {
	return jointPrior->logDensity(hBeta);
}

double CyclicCoordinateDescent::getObjectiveFunction(int convergenceType) {
	if (convergenceType == GRADIENT) {
		real criterion = 0;
		if (useCrossValidation) {
			for (int i = 0; i < K; i++) {
				criterion += hXBeta[i] * hY[i] * hWeights[i];
			}
		} else {
			for (int i = 0; i < K; i++) {
				criterion += hXBeta[i] * hY[i];
			}
		}
		return static_cast<double> (criterion);
	} else
	if (convergenceType == MITTAL) {
		return getLogLikelihood();
	} else
	if (convergenceType == LANGE) {
		return getLogLikelihood() + getLogPrior();
	} else {
    	std::ostringstream stream;
    	stream << "Invalid convergence type: " << convergenceType;
    	error->throwError(stream);	
    }
    return 0.0;
}

double CyclicCoordinateDescent::computeZhangOlesConvergenceCriterion(void) {
	double sumAbsDiffs = 0;
	double sumAbsResiduals = 0;
	if (useCrossValidation) {
		for (int i = 0; i < K; i++) {
			sumAbsDiffs += abs(hXBeta[i] - hXBetaSave[i]) * hWeights[i];
			sumAbsResiduals += abs(hXBeta[i]) * hWeights[i];
		}
	} else {
		for (int i = 0; i < K; i++) {
			sumAbsDiffs += abs(hXBeta[i] - hXBetaSave[i]);
			sumAbsResiduals += abs(hXBeta[i]);
		}
	}
	return sumAbsDiffs / (1.0 + sumAbsResiduals);
}

void CyclicCoordinateDescent::saveXBeta(void) {
	memcpy(hXBetaSave.data(), hXBeta.data(), K * sizeof(real));
}

// struct SerialOnly { };
// struct OpenMP { };
// struct Vanilla { };
// 
// struct C11Threads {
// 	
// 	C11Threads(int threads, size_t size = 100) : nThreads(threads), minSize(size) { }
// 	
// 	int nThreads;
// 	size_t minSize;
// 	
//  };
// 
// namespace variants {
// 
// 	namespace impl {
// 
// 		template <typename InputIt, typename UnaryFunction, class Info>
// 		inline UnaryFunction for_each(InputIt begin, InputIt end, UnaryFunction function, 
// 				Info& info) {
// 			
// 			const int nThreads = info.nThreads;
// 			const size_t minSize = info.minSize;	
// 										
// 			if (nThreads > 1 && std::distance(begin, end) >= minSize) {				  
// 				std::vector<std::thread> workers(nThreads - 1);
// 				size_t chunkSize = std::distance(begin, end) / nThreads;
// 				size_t start = 0;
// 				for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
// 					workers[i] = std::thread(
// 						std::for_each<InputIt, UnaryFunction>,
// 						begin + start, 
// 						begin + start + chunkSize, 
// 						function);
// 				}
// 				auto rtn = std::for_each(begin + start, end, function);
// 				for (int i = 0; i < nThreads - 1; ++i) {
// 					workers[i].join();
// 				}
// 				return rtn;
// 			} else {				
// 				return std::for_each(begin, end, function);
// 			}
// 		}	
// 	
// 	}
// 
//     template <class InputIt, class UnaryFunction, class Specifics>
//     inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, Specifics) {
//         return std::for_each(first, last, f);    
//     }
//     
//     template <class UnaryFunction, class Specifics>
//     inline UnaryFunction for_each(int first, int last, UnaryFunction f, Specifics) {
//         for (; first != last; ++first) {
//             f(first);        
//         }
//         return f;
//     }
//     
//     template <class UnaryFunction>
//     inline UnaryFunction for_each(int first, int last, UnaryFunction f, C11Threads& x) {
//     	return impl::for_each(boost::make_counting_iterator(first), boost::make_counting_iterator(last), f, x);
//     }
//     
// #ifdef OPENMP    
//     template <class UnaryFunction, class Specifics>
//     inline UnaryFunction for_each(int first, int last, UnaryFunction f, OpenMP) {
//         std::cout << "Parallel ";
//         #pragma omp parallel for
//         for (; first != last; ++first) {
//             f(first);        
//         }
//         return f;
//     }        
// #endif     
// }

struct AbstractVariant {

    AbstractVariant(CyclicCoordinateDescent& ccd, 
            AbstractModelSpecifics& modelSpecifics,
            priors::JointPriorPtr& jointPrior,
            std::vector<double>& hBeta,
            std::vector<bool>& fixBeta, 
            std::vector<double>& updates, 
            std::vector<double>& hDelta,
            NoiseLevels noiseLevel) 
            : ccd(ccd), modelSpecifics(modelSpecifics), jointPrior(jointPrior), 
              hBeta(hBeta), fixBeta(fixBeta), updates(updates), hDelta(hDelta), 
              noiseLevel(noiseLevel) { }
                  
protected:

    double updateSingleBeta(int index) {

        if (!ccd.sufficientStatisticsKnown) {
            std::ostringstream stream;
            stream << "Error in state synchronization.";
            ccd.error->throwError(stream);				
        }
        computeNumeratorForGradient(index);
    
        priors::GradientHessian gh;
        computeGradientAndHessian(index, &gh.first, &gh.second);
    
        if (gh.second < 0.0) {
            gh.first = 0.0;	
            gh.second = 0.0;
        }
    
        return jointPrior->getDelta(gh, hBeta, index);
    }
    
    double applyBounds(double inDelta, int index) {
        double delta = inDelta;
        if (delta < -hDelta[index]) {
            delta = -hDelta[index];
        } else if (delta > hDelta[index]) {
            delta = hDelta[index];
        }

        hDelta[index] = max(2.0 * abs(delta), 0.5 * hDelta[index]);
        return delta;
    }    

    virtual void computeNumeratorForGradient(int index) = 0;
    virtual void computeGradientAndHessian(int index, double* gradient, double* hessian) = 0;

    CyclicCoordinateDescent& ccd;
    AbstractModelSpecifics& modelSpecifics;    
    priors::JointPriorPtr& jointPrior;
    std::vector<double>& hBeta;
    std::vector<bool>& fixBeta;  
    std::vector<double>& updates;
    std::vector<double>& hDelta;
    NoiseLevels noiseLevel;   
};

struct CCDVariant : public AbstractVariant {

    using AbstractVariant::AbstractVariant;
    
    void operator()(int index){

        if (!fixBeta[index]) {
            double delta = updateSingleBeta(index);
            delta = applyBounds(delta, index);
            if (delta != 0.0) {
                ccd.sufficientStatisticsKnown = false;
                ccd.updateSufficientStatistics(delta, index);
            }
        }
        
        if((noiseLevel > QUIET) && ((index+1) % 100 == 0)) {
            std::ostringstream stream;
            stream << "Finished variable " << (index+1);
            ccd.logger->writeLine(stream);
        }
	} 
		
	void finalizeUpdate() { } // Do nothing     
	
protected:
    void computeNumeratorForGradient(int index) {
    	modelSpecifics.computeNumeratorForGradient(index);
    }      
    
    void computeGradientAndHessian(int index, double* ogradient, double* ohessian) {    
        modelSpecifics.computeGradientAndHessian(index, ogradient, ohessian, 
            ccd.useCrossValidation);          
    }
};

struct MMVariant : public AbstractVariant {
    
    MMVariant(CyclicCoordinateDescent& ccd,   
            AbstractModelSpecifics& modelSpecifics,
            priors::JointPriorPtr& jointPrior,
            std::vector<double>& hBeta,
            std::vector<double>& hBetaMM,
            std::vector<bool>& fixBeta, 
            std::vector<double>& updates, 
            std::vector<double>& hDelta,
            NoiseLevels noiseLevel) 
            : AbstractVariant(ccd, modelSpecifics, jointPrior, hBeta, fixBeta, updates, 
                    hDelta, noiseLevel), J(ccd.J), scale(1.0), hBetaMM(hBetaMM){
        if (updates.size() != J) {
            updates.resize(J);
        }    
      	modelSpecifics.initializeMM(fixBeta, hBeta, hBetaMM);
      	xbetaduration = 0;
    }
    
    
#define NEW_XBETA
        
    void operator()(int index) {
	    if (!fixBeta[index]) {
			double delta = updateSingleBeta(index);
			delta = applyBounds(delta, index);
#ifndef NEW_XBETA			
			updates[index] = delta;
#else
//          updates[index] = hBeta[index] + delta;
			hBetaMM[index] += delta;
            //hBeta[index] += delta;
#endif		
		}			
	}  
	
	void updateTBB(int index) {
	    if (!fixBeta[index]) {
			double delta = updateSingleBeta(index);
			delta = applyBounds(delta, index);
#ifndef NEW_XBETA			
			updates[index] = delta;
#else
//          updates[index] = hBeta[index] + delta;
			hBetaMM[index] += delta;
            //hBeta[index] += delta;
#endif		
		}			
	}  
		
    const void finalizeUpdate() {   
    	for (int indexCopy = 0; indexCopy < J; indexCopy++) {hBeta[indexCopy] = hBetaMM[indexCopy];}
#ifndef NEW_XBETA     
		for(int index2 = 0; index2 < J; index2 ++) {
			if (updates[index2] != 0.0) {
				ccd.sufficientStatisticsKnown = false;
				ccd.updateSufficientStatistics(updates[index2], index2);
			}
		}  
#else
       modelSpecifics.computeXBeta(&hBeta[0]);		 

#endif
        ccd.computeRemainingStatistics(true,0);       
    } 
    
    void finalizeUpdateParallel(){
        for (int indexCopy = 0; indexCopy < J; indexCopy++) {hBeta[indexCopy] = hBetaMM[indexCopy];}
		gettimeofday(&time1, NULL);
        modelSpecifics.computeXBeta(&hBeta[0]);
    	gettimeofday(&time2, NULL);
		xbetaduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	
        ccd.computeRemainingStatistics(true,0);       
    }

    
    void finalizeUpdateParallel(C11Threads & parallelScheme){
        for (int indexCopy = 0; indexCopy < J; indexCopy++) {hBeta[indexCopy] = hBetaMM[indexCopy];}
		gettimeofday(&time1, NULL);
        modelSpecifics.computeXBeta(&hBeta[0], parallelScheme);
    	gettimeofday(&time2, NULL);
		xbetaduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	
        ccd.computeRemainingStatistics(true,0);       
    }

    
    void finalizeUpdateParallel(C11ThreadPool & parallelScheme){
        for (int indexCopy = 0; indexCopy < J; indexCopy++) {hBeta[indexCopy] = hBetaMM[indexCopy];}
		gettimeofday(&time1, NULL);
        modelSpecifics.computeXBeta(&hBeta[0], parallelScheme);
    	gettimeofday(&time2, NULL);
		xbetaduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	
        ccd.computeRemainingStatistics(true,0);       
    }
    
    void setScale(double s) { scale = s; }
    
    void mmTime(){
    std::cout << "xbeta time = " << xbetaduration << std::endl;
    }
    
protected: 

    void computeNumeratorForGradient(int index) { }      
    
    void computeGradientAndHessian(int index, double* ogradient, double* ohessian) {    
        modelSpecifics.computeMMGradientAndHessian(index, ogradient, ohessian,  scale,
            ccd.useCrossValidation);          
    }    
    
private:
    size_t J;     
    double scale;    
    std::vector<double>& hBetaMM;      
    struct timeval time1, time2;  
    double xbetaduration;      
};

void CyclicCoordinateDescent::update(
		int maxIterations,
		int convergenceType,
		double epsilon
		) {

	if (convergenceType < GRADIENT || convergenceType > ZHANG_OLES) {
	    std::ostringstream stream;
		stream << "Unknown convergence criterion: " << convergenceType;
		error->throwError(stream);				
	}

	if (!validWeights) {    	   	
		computeNEvents();
		computeFixedTermsInLogLikelihood();
		computeFixedTermsInGradientAndHessian();
		validWeights = true;
	}

	if (!xBetaKnown) {
		computeXBeta();
		xBetaKnown = true;
		sufficientStatisticsKnown = false;
	}

	if (!sufficientStatisticsKnown) {
		computeRemainingStatistics(true, 0); // TODO Check index?
		sufficientStatisticsKnown = true;
	}

	resetBounds();

	bool done = false;
	int iteration = 0;
	double lastObjFunc = 0.0;

	if (convergenceType < ZHANG_OLES) {
		lastObjFunc = getObjectiveFunction(convergenceType);
	} else { // ZHANG_OLES
		saveXBeta();
	}
	
	double betaUpdaterScale = 1.0; 
	int nThreads = 4;
	int mmIndex1Limit = 1;
	int mmIndex2Limit = 1;
	int qnQ = 4;


//#define noMM
//#define quasiNewton
#ifdef noMM
    auto betaUpdater = CCDVariant(*this, modelSpecifics, jointPrior, hBeta, fixBeta, 
                            hUpdates, hDelta, noiseLevel);
    auto parallelScheme = Vanilla();
    cout << "noMM" << endl;
    cout << "betaUpdaterScale = " << betaUpdaterScale << endl;
    cout << "nThreads = " << nThreads << endl;
#else                            
    auto betaUpdater = MMVariant(*this, modelSpecifics, jointPrior, hBeta, hBetaMM, fixBeta, 
                            hUpdates, hDelta, noiseLevel); 
     
    betaUpdater.setScale(betaUpdaterScale);          
    //auto parallelScheme = Vanilla();
    
    cout << "MM" << endl;
    cout << "betaUpdaterScale = " << betaUpdaterScale << endl;
    cout << "nThreads = " << nThreads << endl;

	//auto parallelScheme = C11Threads(nThreads);
	//auto parallelSchemeXBeta = C11Threads(nThreads);
 	C11ThreadPool parallelScheme(nThreads,nThreads);
 	C11ThreadPool parallelSchemeXBeta(nThreads,nThreads);
#endif

    struct timeval time1, time2;  
    double parallelforduration;      

                    
    double thisLogPost = 0; 
    double lastLogPost = 0; 
    
             		
#ifdef quasiNewton
    	
	    using namespace Eigen;

	    Eigen::MatrixXd secantsU(J, qnQ);
	    Eigen::MatrixXd secantsV(J, qnQ);

	    VectorXd x(J);

	    // Fill initial secants
	    int countU = 0;
	    int countV = 0;

	    for (int q = 0; q < qnQ; ++q) {
	        x = Map<const VectorXd>(hBeta.data(), J); // Make copy

			gettimeofday(&time1, NULL);
			variants::for_each(0, J, betaUpdater, parallelScheme);	
			gettimeofday(&time2, NULL);
			parallelforduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	
	
			//betaUpdater.finalizeUpdateParallel(parallelSchemeXBeta);
			betaUpdater.finalizeUpdateParallel(parallelSchemeXBeta);

            if (countU == 0) { // First time through
                secantsU.col(countU) = Map<const VectorXd>(hBeta.data(), J) - x;
                ++countU;
            } else if (countU < qnQ - 1) { // Middle case
                secantsU.col(countU) = Map<const VectorXd>(hBeta.data(), J) - x;
                secantsV.col(countV) = secantsU.col(countU);
                ++countU;
                ++countV;
            } else { // Last time through
                secantsV.col(countV) = Map<const VectorXd>(hBeta.data(), J) - x;
                ++countV;
            }
	    }  
	    //exit(-1);     
#endif  		
	while (!done) {
	
#ifdef quasiNewton

	    int newestSecant = qnQ - 1;
	    int previousSecant = newestSecant - 1;
		// 2 cycles for each QN step
	    
	    x = Map<const VectorXd>(hBeta.data(), J); // Make copy
	    
	    // cycle
	    gettimeofday(&time1, NULL);
		variants::for_each(0, J, betaUpdater, parallelScheme);
		
		gettimeofday(&time2, NULL);
		parallelforduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	
		
		
		//betaUpdater.finalizeUpdateParallel(parallelSchemeXBeta);
		betaUpdater.finalizeUpdateParallel();
		
		VectorXd Fx = Map<const VectorXd>(hBeta.data(), J); // TODO Can remove?   
	    
	    secantsU.col(newestSecant) = Fx - x;

        // cycle
	    gettimeofday(&time1, NULL);
		variants::for_each(0, J, betaUpdater, parallelScheme);
		gettimeofday(&time2, NULL);
		parallelforduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	

        //betaUpdater.finalizeUpdateParallel(parallelSchemeXBeta);
        betaUpdater.finalizeUpdateParallel();
        
        secantsV.col(newestSecant) = Map<const VectorXd>(hBeta.data(), J) - Fx;
    
        auto M = secantsU.transpose() * (secantsU - secantsV);

        auto Minv = M.inverse();

        auto A = secantsU.transpose() * secantsU.col(newestSecant);
        auto B = Minv * A;
        auto C = secantsV * B;
     	VectorXd xqn = Fx + C; // TODO Can remove?

        // Save CCD solution
        x = Map<const VectorXd>(hBeta.data(), J);
        
        double ccdObjective = getLogLikelihood() + getLogPrior();
        
		std::cout << "2 steps ccdObjective = " << ccdObjective << endl;
		
        Map<VectorXd>(hBeta.data(), J) = xqn; // Set QN solution
        modelSpecifics.computeXBeta(&hBeta[0], parallelScheme);
		computeRemainingStatistics(true,0);

        double qnObjective = getLogLikelihood() + getLogPrior();
		std::cout << "secant ccdObjective = " << qnObjective << endl;

    	if (ccdObjective > qnObjective) { // Revert
            Map<VectorXd>(hBeta.data(), J) = x; // Set CCD solution
        	modelSpecifics.computeXBeta(&hBeta[0], parallelScheme);
			computeRemainingStatistics(true,0);

        	double ccd2Objective = getLogLikelihood() + getLogPrior();

            if (ccdObjective != ccd2Objective) {
                std::cerr << "Poor revert: " << ccdObjective << " != " << ccd2Objective << " diff: "<< (ccdObjective - ccd2Objective) << std::endl;
            }
            // lastObjFunc = savedLastObjFunc;
            std::cerr << "revert" << std::endl;
        } else {
            std::cerr << "accept" << std::endl;
        }

        //done = check();
        previousSecant = newestSecant;
        newestSecant = (newestSecant + 1) % qnQ;
#endif

#ifndef quasiNewton

 	gettimeofday(&time1, NULL);
	tbb::parallel_for(0, J, 1, [=](int i) {const_cast<MMVariant*>(&betaUpdater)->updateTBB(i);});
	gettimeofday(&time2, NULL);
	parallelforduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	

/*   
	for (int mmIndex1 = 0; mmIndex1 < mmIndex1Limit; mmIndex1++){
	    gettimeofday(&time1, NULL);
		variants::for_each(0, J, betaUpdater, parallelScheme);
		gettimeofday(&time2, NULL);
		parallelforduration += time2.tv_sec - time1.tv_sec + (double)(time2.tv_usec - time1.tv_usec) / 1000000.0;	

	}
*/	

#endif
	 
	//betaUpdater.finalizeUpdateParallel(parallelSchemeXBeta);
	betaUpdater.finalizeUpdateParallel();
		
#ifndef noMM
	    if (lastLogPost > thisLogPost && iteration != 1){
	 		betaUpdaterScale = max(betaUpdaterScale / 1.0,1.0) + 8*iteration*iteration;
	 		betaUpdater.setScale(betaUpdaterScale);  
	    }
#endif

		if (iteration == 1){
			//exit(-1); 
		}

		iteration++;
//		bool checkConvergence = (iteration % J == 0 || iteration == maxIterations);
		bool checkConvergence = true; // Check after each complete cycle

		if (checkConvergence) {

			double conv;
			bool illconditioned = false;
			if (convergenceType < ZHANG_OLES) {
 				double thisObjFunc = getObjectiveFunction(convergenceType);
				if (thisObjFunc != thisObjFunc) {
				    std::ostringstream stream;
					stream << "\nWarning! problem is ill-conditioned for this choice of hyperparameter. Enforcing convergence!";
					logger->writeLine(stream);
					conv = 0.0;
					illconditioned = true;
				} else {
					conv = computeConvergenceCriterion(thisObjFunc, lastObjFunc);
				}
				lastObjFunc = thisObjFunc;
			} else { // ZHANG_OLES
				conv = computeZhangOlesConvergenceCriterion();
				saveXBeta();
			} // Necessary to call getObjFxn or computeZO before getLogLikelihood,
			  // since these copy over XBeta

			lastLogPost = thisLogPost;
			double thisLogLikelihood = getLogLikelihood();
			double thisLogPrior = getLogPrior();
			thisLogPost = thisLogLikelihood + thisLogPrior;
            std::ostringstream stream;
			if (noiseLevel > QUIET) {			    
				stream << "\n";				
				printVector(&hBeta[0], J, stream);
				stream << "\n";
				stream << "log post: " << thisLogPost
						<< " (" << thisLogLikelihood << " + " << thisLogPrior
						<< ") (iter:" << iteration << ") ";
			}

			if (epsilon > 0 && conv < epsilon) {
				if (illconditioned) {
					lastReturnFlag = ILLCONDITIONED;
				} else {
					if (noiseLevel > SILENT) {
						stream << "Reached convergence criterion";
					}
					lastReturnFlag = SUCCESS;
				}
				done = true;
			} else if (iteration == maxIterations) {
				if (noiseLevel > SILENT) {
					stream << "Reached maximum iterations";
				}
				done = true;
				lastReturnFlag = MAX_ITERATIONS;
			}
			if (noiseLevel > QUIET) {
                logger->writeLine(stream);
			}
			logger->yield();			
		}						
	}
	lastIterationCount = iteration;
	updateCount += 1;
	
	betaUpdater.mmTime();
	cout << "parallel for time = " << parallelforduration << std::endl;

	fisherInformationKnown = false;
	varianceKnown = false;
}

/**
 * Computationally heavy functions
 */

void CyclicCoordinateDescent::computeGradientAndHessian(int index, double *ogradient,
		double *ohessian) {
	modelSpecifics.computeGradientAndHessian(index, ogradient, ohessian, useCrossValidation);
}

void CyclicCoordinateDescent::computeNumeratorForGradient(int index) {
	// Delegate
	modelSpecifics.computeNumeratorForGradient(index);
}

void CyclicCoordinateDescent::computeRatiosForGradientAndHessian(int index) {
    std::ostringstream stream;
	stream << "Error!";
	error->throwError(stream);
}

double CyclicCoordinateDescent::getHessianDiagonal(int index) {

	checkAllLazyFlags();
	double g_d1, g_d2;

	computeNumeratorForGradient(index);
	computeGradientAndHessian(index, &g_d1, &g_d2);

	return g_d2;
}

double CyclicCoordinateDescent::getAsymptoticVariance(int indexOne, int indexTwo) {
	checkAllLazyFlags();
	if (!fisherInformationKnown) {
		computeAsymptoticPrecisionMatrix();
		fisherInformationKnown = true;
	}

	if (!varianceKnown) {
		computeAsymptoticVarianceMatrix();
		varianceKnown = true;
	}

	IndexMap::iterator itOne = hessianIndexMap.find(indexOne);
	IndexMap::iterator itTwo = hessianIndexMap.find(indexTwo);

	if (itOne == hessianIndexMap.end() || itTwo == hessianIndexMap.end()) {
		return NAN;
	} else {
		return varianceMatrix(itOne->second, itTwo->second);
	}
}

double CyclicCoordinateDescent::getAsymptoticPrecision(int indexOne, int indexTwo) {
	checkAllLazyFlags();
	if (!fisherInformationKnown) {
		computeAsymptoticPrecisionMatrix();
		fisherInformationKnown = true;
	}

	IndexMap::iterator itOne = hessianIndexMap.find(indexOne);
	IndexMap::iterator itTwo = hessianIndexMap.find(indexTwo);

	if (itOne == hessianIndexMap.end() || itTwo == hessianIndexMap.end()) {
		return NAN;
	} else {
		return hessianMatrix(itOne->second, itTwo->second);
	}
}

CyclicCoordinateDescent::Matrix CyclicCoordinateDescent::computeFisherInformation(const std::vector<size_t>& indices) const {
    Matrix fisherInformation(indices.size(), indices.size());
    for (size_t ii = 0; ii < indices.size(); ++ii) {
        const int i = indices[ii];
        for (size_t jj = ii; jj < indices.size(); ++jj) {
            const int j = indices[jj];
            double element = 0.0;            
             modelSpecifics.computeFisherInformation(i, j, &element, useCrossValidation);
            fisherInformation(jj, ii) = fisherInformation(ii, jj) = element;
        }
    }
    return fisherInformation;
}

void CyclicCoordinateDescent::computeAsymptoticPrecisionMatrix(void) {

	typedef std::vector<int> int_vec;
	int_vec indices;
	hessianIndexMap.clear();

	int index = 0;
	for (int j = 0; j < J; ++j) {
		if (!fixBeta[j] &&
				(priorType != priors::LAPLACE || getBeta(j) != 0.0)) {
			indices.push_back(j);
			hessianIndexMap[j] = index;
			index++;
		}
	}

	hessianMatrix.resize(indices.size(), indices.size());
	modelSpecifics.makeDirty(); // clear hessian terms

	for (size_t ii = 0; ii < indices.size(); ++ii) {
		for (size_t jj = ii; jj < indices.size(); ++jj) {
			const int i = indices[ii];
			const int j = indices[jj];
//			std::cerr << "(" << i << "," << j << ")" << std::endl;
			double fisherInformation = 0.0;
			modelSpecifics.computeFisherInformation(i, j, &fisherInformation, useCrossValidation);
//			if (fisherInformation != 0.0) {
				// Add tuple to sparse matrix
//				tripletList.push_back(Triplet<double>(ii,jj,fisherInformation));
//			}
			hessianMatrix(jj,ii) = hessianMatrix(ii,jj) = fisherInformation;

//			std::cerr << "Info = " << fisherInformation << std::endl;
//			std::cerr << "Hess = " << getHessianDiagonal(0) << std::endl;
//			exit(-1);
		}
	}

//	sm.setFromTriplets(tripletList.begin(), tripletList.end());
//	cout << sm;
//	auto inv = sm.triangularView<Upper>().solve(dv1);
//	cout << sm.inverse();
//	cout << hessianMatrix << endl;

	// Take inverse
//	hessianMatrix = hessianMatrix.inverse();

//	cout << hessianMatrix << endl;
}

void CyclicCoordinateDescent::computeAsymptoticVarianceMatrix(void) {
	varianceMatrix = hessianMatrix.inverse();
// 	cout << varianceMatrix << endl;
}

// double CyclicCoordinateDescent::ccdUpdateBeta(int index) {
// 
// 	if (!sufficientStatisticsKnown) {
// 	    std::ostringstream stream;
// 		stream << "Error in state synchronization.";
// 		error->throwError(stream);				
// 	}
// 		
// 	computeNumeratorForGradient(index);
// 	
// 	priors::GradientHessian gh;
// 	computeGradientAndHessian(index, &gh.first, &gh.second);
// 	
// 	if (gh.second < 0.0) {
// 	    gh.first = 0.0;	
// 	    gh.second = 0.0;
// 	}
// 	
//     return jointPrior->getDelta(gh, hBeta, index);
// }

template <class IteratorType>
void CyclicCoordinateDescent::axpy(real* y, const real alpha, const int index) {
	IteratorType it(*hXI, index);
	for (; it; ++it) {
		const int k = it.index();
		y[k] += alpha * it.value();
	}
}

void CyclicCoordinateDescent::axpyXBeta(const real beta, const int j) {
	if (beta != static_cast<real>(0.0)) {
		switch (hXI->getFormatType(j)) {
		case INDICATOR:
			axpy < IndicatorIterator > (hXBeta.data(), beta, j);
			break;
		case DENSE:
			axpy < DenseIterator > (hXBeta.data(), beta, j);
			break;
		case SPARSE:
			axpy < SparseIterator > (hXBeta.data(), beta, j);
			break;
		default:
			// throw error
			std::ostringstream stream;
			stream << "Unknown vector type.";
			error->throwError(stream);
		}
	}
}

void CyclicCoordinateDescent::computeXBeta(void) {
	// Note: X is current stored in (sparse) column-major format, which is
	// inefficient for forming X\beta.
	// TODO Make row-major version of X

	if (setBetaList.empty()) { // Update all
		// clear X\beta
		zeroVector(hXBeta.data(), K);
		for (int j = 0; j < J; ++j) {
			axpyXBeta(hBeta[j], j);
		}
	} else {
		while (!setBetaList.empty()) {
			SetBetaEntry entry = setBetaList.front();
			axpyXBeta(hBeta[entry.first] - entry.second, entry.first);
			setBetaList.pop_front();
		}
	}
}

void CyclicCoordinateDescent::updateXBeta(double delta, int index) {
	// Update beta
	real realDelta = static_cast<real>(delta);
	hBeta[index] += delta;

	// Delegate
	modelSpecifics.updateXBeta(realDelta, index, useCrossValidation);
}

void CyclicCoordinateDescent::updateSufficientStatistics(double delta, int index) {
	updateXBeta(delta, index);
	sufficientStatisticsKnown = true;
}

void CyclicCoordinateDescent::computeRemainingStatistics(bool allStats, int index) { // TODO Rename
	// Separate function for benchmarking
	if (allStats) {
		// Delegate
		modelSpecifics.computeRemainingStatistics(useCrossValidation);
	}
}

/**
 * Updating and convergence functions
 */

double CyclicCoordinateDescent::computeConvergenceCriterion(double newObjFxn, double oldObjFxn) {	
	// This is the stopping criterion that Ken Lange generally uses
	return abs(newObjFxn - oldObjFxn) / (abs(newObjFxn) + 1.0);
}

// double CyclicCoordinateDescent::applyBounds(double inDelta, int index) {
// 	double delta = inDelta;
// 	if (delta < -hDelta[index]) {
// 		delta = -hDelta[index];
// 	} else if (delta > hDelta[index]) {
// 		delta = hDelta[index];
// 	}
// 
// 	hDelta[index] = max(2.0 * abs(delta), 0.5 * hDelta[index]);
// 	return delta;
// }

/**
 * Utility functions
 */

void CyclicCoordinateDescent::computeFixedTermsInLogLikelihood(void) {
	modelSpecifics.computeFixedTermsInLogLikelihood(useCrossValidation);
}

void CyclicCoordinateDescent::computeFixedTermsInGradientAndHessian(void) {
	// Delegate
	modelSpecifics.computeFixedTermsInGradientAndHessian(useCrossValidation);
}

template <class T>
void CyclicCoordinateDescent::printVector(T* vector, int length, ostream &os) {
	os << "(" << vector[0];
	for (int i = 1; i < length; i++) {
		os << ", " << vector[i];
	}
	os << ")";			
}

template <class T> 
T* CyclicCoordinateDescent::readVector(
		const char *fileName,
		int *length) {

	ifstream fin(fileName);
	T d;
	std::vector<T> v;
	
	while (fin >> d) {
		v.push_back(d);
	}
	
	T* ptr = (T*) malloc(sizeof(T) * v.size());
	for (int i = 0; i < v.size(); i++) {
		ptr[i] = v[i];
	}
		
	*length = v.size();
	return ptr;
}

void CyclicCoordinateDescent::testDimension(int givenValue, int trueValue, const char *parameterName) {
	if (givenValue != trueValue) {
	    std::ostringstream stream;
		stream << "Wrong dimension in " << parameterName << " vector.";
		error->throwError(stream);		
	}
}	

inline int CyclicCoordinateDescent::sign(double x) {
	if (x == 0) {
		return 0;
	}
	if (x < 0) {
		return -1;
	}
	return 1;
}

} // namespace
