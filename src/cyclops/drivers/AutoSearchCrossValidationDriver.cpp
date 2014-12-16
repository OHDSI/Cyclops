/*
 * AutoSearchCrossValidationDriver.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: msuchard
 */


// TODO Change from fixed grid to adaptive approach in BBR

#include <iostream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <cstdlib>
#include <iterator>

#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
//#include "ccd.h"
#include "../utils/HParSearch.h"
#include "engine/ThreadPool.h"

#include "boost/iterator/counting_iterator.hpp"

#include <Rcpp.h>

namespace bsccs {

const static int MAX_STEPS = 50;

AutoSearchCrossValidationDriver::AutoSearchCrossValidationDriver(
			const ModelData& _modelData,
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,			
			loggers::ProgressLoggerPtr _logger,
			loggers::ErrorHandlerPtr _error,			
            vector<real>* wtsExclude			
			) : AbstractCrossValidationDriver(_logger, _error), modelData(_modelData), maxPoint(0), gridSize(iGridSize),
			lowerLimit(iLowerLimit), upperLimit(iUpperLimit), weightsExclude(wtsExclude),
			maxSteps(MAX_STEPS) {

	// Do anything???
}

AutoSearchCrossValidationDriver::~AutoSearchCrossValidationDriver() {
	// Do nothing
}

double AutoSearchCrossValidationDriver::computeGridPoint(int step) {
	if (gridSize == 1) {
		return upperLimit;
	}
	// Log uniform grid
	double stepSize = (log(upperLimit) - log(lowerLimit)) / (gridSize - 1);
	return exp(log(lowerLimit) + step * stepSize);
}

void AutoSearchCrossValidationDriver::logResults(const CCDArguments& arguments) {

	ofstream outLog(arguments.cvFileName.c_str());
	if (!outLog) {
	    std::ostringstream stream;
		stream << "Unable to open log file: " << arguments.cvFileName;
		error->throwError(stream);		
	}
	outLog << std::scientific << maxPoint << std::endl;
	outLog.close();
}

void AutoSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
		const CCDArguments& arguments) {

	ccd.setWeights(NULL);
	ccd.setHyperprior(maxPoint);
	ccd.resetBeta(); // Cold-start
}

template <typename InputIt, typename UnaryFunction>
inline UnaryFunction for_each_thread(InputIt begin, InputIt end, UnaryFunction function, 
		const int nThreads) {	
	std::vector<std::thread> workers(nThreads - 1);
	size_t chunkSize = std::distance(begin, end) / nThreads;
	size_t start = 0;
	for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
		workers[i] = std::thread(
			std::for_each<InputIt, UnaryFunction>,
			begin + start, 
			begin + start + chunkSize, 
			function);
	}
	auto rtn = std::for_each(begin + start, end, function);
	for (int i = 0; i < nThreads - 1; ++i) {
		workers[i].join();
	}
	return rtn;	
}

double AutoSearchCrossValidationDriver::doCrossValidation(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments,
		int step,
		std::vector<double> & predLogLikelihood){


	std::vector<real> weights;

	predLogLikelihood.resize(arguments.foldToCompute);

#if 1		
	
	std::cerr << "Threads = " << arguments.threads << std::endl;
	int threads = (arguments.threads > 1) ? arguments.threads : 2;
	std::cerr << "Threads = " << threads << std::endl;
	
	auto start1 = std::chrono::steady_clock::now();
		
	
	std::vector<CyclicCoordinateDescent*> ccdPool;
	std::vector<AbstractSelector*> selectorPool;
	
// 	ccdPool.push_back(&ccd);
// 	selectorPool.push_back(&selector);
		
	for (int i = 0 /* 1 */; i < arguments.foldToCompute; ++i) {
		ccdPool.push_back(new CyclicCoordinateDescent(ccd));
		selectorPool.push_back(selector.clone());
	}
		    
#define PARA    

	auto& weightsExclude = this->weightsExclude;
	auto& logger = this->logger;
    
#ifdef PARA    

//#define POOL
#ifdef POOL
	ThreadPool pool(1);	
    std::vector< std::future<void> > results; 
	for (int task = 0; task < arguments.foldToCompute; ++task) {	    
        results.emplace_back(
		    pool.enqueue(
#else // POOL
	
	std::mutex lock;
	
	int nThreads = 2;
		
	for_each_thread(boost::make_counting_iterator(0), boost::make_counting_iterator(arguments.foldToCompute),

#endif // POOL
#else // PARA
	std::for_each(boost::make_counting_iterator(0), boost::make_counting_iterator(arguments.foldToCompute),
#endif // PARA		    

            [step, &ccdPool, &selectorPool, 
            &arguments, &predLogLikelihood, 
            	&weightsExclude, &logger , &lock
			 //    ,&ccd, &selector
			    ](int task) {
	
					auto uniqueId = int{0};
					    
//				    std::cerr << "Running #" << task << " with id " << uniqueId << std::endl;
				    auto& ccdTask = *ccdPool[task]; // Task-specific // TODO MAKE THREAD-SPECIFIC
				    auto& selectorTask = *selectorPool[task]; // Task-specific // TODO MAKE THREAD-SPECIFIC

// 					auto ccdTaskPtr = bsccs::unique_ptr<CyclicCoordinateDescent>(new CyclicCoordinateDescent(ccd));
// 					auto selectorTaskPtr = bsccs::unique_ptr<AbstractSelector>(selector.clone());
// 					auto& ccdTask = *ccdTaskPtr;
// 					auto& selectorTask = *selectorTaskPtr;
//   				    				     				    				    
// 				    // Bring selector up-to-date
// 				    selectorTask.reseed();
// 				    for (int i = 0; i <= task; ++i) {
// 				    	int fold = i % arguments.fold;
// 				    	if (fold == 0) {
// 				    		selectorTask.permute();
// 				    	}
// 				    }
// 				    
// 				    int fold = task % arguments.fold;
// 				    				    
// 					// Get this fold and update
// 					std::vector<real> weights; // Task-specific
// 					selectorTask.getWeights(fold, weights);
// 					if (weightsExclude){
// 						for(auto j = 0; j < weightsExclude->size(); j++){
// 							if (weightsExclude->at(j) == 1.0){
// 								weights[j] = 0.0;
// 							}
// 						}
// 					}
// 					ccdTask.setWeights(&weights[0]);
// 					std::ostringstream stream;
// 					stream << "Running at " << ccdTask.getPriorInfo() << " ";

					ccdTask.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);
					
					lock.lock();
					std::cerr << "ccd " << task << " @ " << &ccdTask << std::endl;
					lock.unlock();
					
					
 					
				//	lock.unlock();

					// Compute predictive loglikelihood for this fold
// 					selectorTask.getComplement(weights);  // TODO THREAD_SAFE
// 					if (weightsExclude){
// 						for(int j = 0; j < (int)weightsExclude->size(); j++){
// 							if(weightsExclude->at(j) == 1.0){
// 								weights[j] = 0.0;
// 							}
// 						}
// 					}
// 
// 					double logLikelihood = ccdTask.getPredictiveLogLikelihood(&weights[0]);
// 
// 					stream << "Grid-point #" << (step + 1) << " at "; // << ccd.getHyperprior();
// 					std::vector<double> hyperprior = ccdTask.getHyperprior();
// 					std::copy(hyperprior.begin(), hyperprior.end(),
// 						std::ostream_iterator<double>(stream, " "));
// 		
// 					stream << "\tFold #" << (fold + 1)
// 							  << " Rep #" << (task / arguments.fold + 1) << " pred log like = "
// 							  << logLikelihood;
// 					logger->writeLine(stream);				  
// 
// 					// Store value
// 					predLogLikelihood[task] = logLikelihood;				    				    				    				    				    				    
			    }
		    
#ifdef PARA		   
#ifdef POOL 
		    )
		);
	}
	for (auto&& result: results) result.get();	
#else // POOL

#endif // POOL
	, nThreads);
#else // PARA
	);
#endif // PARA

	auto end1 = std::chrono::steady_clock::now();	
	
	std::cerr << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count()	
			  << std::endl;
	
	
	
	// Clean up
// 	for (int i = 1; i < arguments.foldToCompute; ++i) {
// 		std::cerr << "Deleting extra copies at " << i << std::endl;
// 		delete ccdPool[i];
// 		delete selectorPool[i];		
// 	}	
	
// 	std::cerr << &ccd.getHyperprior() << std::endl;
// 	std::cerr << &ccdCopy.getHyperprior() << std::endl;
#endif	

	auto start2 = std::chrono::steady_clock::now();	

	selector.reseed();
	
	/* start code duplication */
	//std::vector<double> predLogLikelihood;
	for (int i = 0; i < arguments.foldToCompute; i++) {
		int fold = i % arguments.fold;
		if (fold == 0) {
			selector.permute(); // Permute every full cross-validation rep
		}

		// Get this fold and update
		selector.getWeights(fold, weights);
		if(weightsExclude){
			for(int j = 0; j < (int)weightsExclude->size(); j++){
				if(weightsExclude->at(j) == 1.0){
					weights[j] = 0.0;
				}
			}
		}
		ccd.setWeights(&weights[0]);
		std::ostringstream stream;
		stream << "Running at " << ccd.getPriorInfo() << " ";
		
		
		ccd.resetBeta(); // TODO REMOVE
		 // TODO THREAD-SPECIFIC
		ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);

		// Compute predictive loglikelihood for this fold
		selector.getComplement(weights); 
		if(weightsExclude){
			for(int j = 0; j < (int)weightsExclude->size(); j++){
				if(weightsExclude->at(j) == 1.0){
					weights[j] = 0.0;
				}
			}
		}

		double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]); 

		stream << "Grid-point #" << (step + 1) << " at "; // << ccd.getHyperprior();
		std::vector<double> hyperprior = ccd.getHyperprior();
		std::copy(hyperprior.begin(), hyperprior.end(),
		    std::ostream_iterator<double>(stream, " "));
		
		stream << "\tFold #" << (fold + 1)
				  << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
				  << logLikelihood;
        logger->writeLine(stream);				  

		// Store value
		predLogLikelihood.push_back(logLikelihood); // TODO THREAD-SAFE
	}
	
	auto end2 = std::chrono::steady_clock::now();	
	
	std::cerr << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count()	
			  << std::endl;	

	double pointEstimate = computePointEstimate(predLogLikelihood);
	/* end code duplication */
	
	::Rf_error("c++ exception (unknown reason)"); 		

	return(pointEstimate);

}

void AutoSearchCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Check that selector is type of CrossValidationSelector

	double tryvalue = modelData.getNormalBasedDefaultVar();
	UniModalSearch searcher(10, 0.01, log(1.5));
//	const double eps = 0.05; //search stopper
	std::ostringstream stream;
	stream << "Default var = " << tryvalue;
	logger->writeLine(stream);

	bool finished = false;

	int step = 0;
	while (!finished) {
		ccd.setHyperprior(tryvalue);
		selector.reseed();		

		std::vector<double> predLogLikelihood;

		// Newly re-located code
		double pointEstimate = doCrossValidation(ccd, selector, arguments, step, predLogLikelihood);

		double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

        std::ostringstream stream;
				stream << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
        searcher.tried(tryvalue, pointEstimate, stdDevEstimate);
        pair<bool,double> next = searcher.step();
        stream << "Completed at " << tryvalue << std::endl;
        stream << "Next point at " << next.second << " and " << next.first;
        logger->writeLine(stream);

        tryvalue = next.second;
        if (!next.first) {
            finished = true;
        }
        std::ostringstream stream1;
        stream1 << searcher;
        logger->writeLine(stream1);
        step++;
        if (step >= maxSteps) {
          std::ostringstream stream;
        	stream << "Max steps reached!";
        	logger->writeLine(stream);
        	finished = true;
        }
	}

	maxPoint = tryvalue;

	// Report results
	std::ostringstream stream1;
	stream1 << std::endl;
	stream1 << "Maximum predicted log likelihood estimated at:" << std::endl;
	stream1 << "\t" << maxPoint << " (variance)" << std::endl;
	if (!arguments.useNormalPrior) {
		double lambda = convertVarianceToHyperparameter(maxPoint);
		stream1 << "\t" << lambda << " (lambda)" << std::endl;
	}	
	logger->writeLine(stream1);
}

} // namespace
