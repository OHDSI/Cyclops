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
#include <thread>

#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
//#include "ccd.h"
#include "../utils/HParSearch.h"

#include "boost/iterator/counting_iterator.hpp"

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

template <typename InputIt>
struct TaskScheduler {

	TaskScheduler(InputIt begin, InputIt end, int nThreads) 
	   : begin(begin), end(end), nThreads(nThreads), 
	     chunkSize(
	     	std::distance(begin, end) / nThreads + (std::distance(begin, end) % nThreads != 0)
	     ) { }
	     
	template <typename UnaryFunction>
	UnaryFunction execute(UnaryFunction function) {	
		
		std::vector<std::thread> workers(nThreads - 1);		
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
	
	size_t getThreadIndex(size_t i) {
		return nThreads == 1 ? 0 :
			i / chunkSize;
	}	
	
private:
	const InputIt begin;
	const InputIt end;
	const int nThreads;
	const size_t chunkSize;
};

double AutoSearchCrossValidationDriver::doCrossValidation(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments,
		int step,
		bool coldStart,
		int nThreads,
		std::vector<CyclicCoordinateDescent*>& ccdPool,
		std::vector<AbstractSelector*>& selectorPool,		
		std::vector<double>& predLogLikelihood){


	predLogLikelihood.resize(arguments.foldToCompute);

// #define NEW_LOOP
// 
// #ifdef NEW_LOOP	
//     std::cerr << "NEW_LOOP" << std::endl;	
// 	auto start1 = std::chrono::steady_clock::now();
				
	auto& weightsExclude = this->weightsExclude;
	auto& logger = this->logger;
	
	auto scheduler = TaskScheduler<decltype(boost::make_counting_iterator(0))>(
		boost::make_counting_iterator(0), 
		boost::make_counting_iterator(arguments.foldToCompute),		
		nThreads);	
			
	auto oneTask =
		[step, coldStart, nThreads, &ccdPool, &selectorPool, 
		&arguments, &predLogLikelihood, 
			&weightsExclude, &logger //, &lock
		 //    ,&ccd, &selector
		 		, &scheduler
			](int task) {
			
				auto ccdTask = ccdPool[scheduler.getThreadIndex(task)];
				auto selectorTask = selectorPool[scheduler.getThreadIndex(task)];
																			
				// Bring selector up-to-date
				if (task == 0 || nThreads > 1) {
    				selectorTask->reseed();				
    			}
    			int i = (nThreads == 1) ? task : 0;    			
				for ( ; i <= task; ++i) {
					int fold = i % arguments.fold;
					if (fold == 0) {
						selectorTask->permute();
					}
				}
				
				int fold = task % arguments.fold;
									
				// Get this fold and update
				std::vector<real> weights; // Task-specific
				selectorTask->getWeights(fold, weights);
				if (weightsExclude){
					for(auto j = 0; j < weightsExclude->size(); j++){
						if (weightsExclude->at(j) == 1.0){
							weights[j] = 0.0;
						}
					}
				}
				ccdTask->setWeights(&weights[0]);
				std::ostringstream stream;
				stream << "Running at " << ccdTask->getPriorInfo() << " ";
				
				if (coldStart) ccdTask->resetBeta();

				ccdTask->update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);
				
				// Compute predictive loglikelihood for this fold
				selectorTask->getComplement(weights);  // TODO THREAD_SAFE
				if (weightsExclude){
					for(int j = 0; j < (int)weightsExclude->size(); j++){
						if(weightsExclude->at(j) == 1.0){
							weights[j] = 0.0;
						}
					}
				}

				double logLikelihood = ccdTask->getPredictiveLogLikelihood(&weights[0]);

				stream << "Grid-point #" << (step + 1) << " at "; // << ccd.getHyperprior();
				std::vector<double> hyperprior = ccdTask->getHyperprior();
				std::copy(hyperprior.begin(), hyperprior.end(),
					std::ostream_iterator<double>(stream, " "));
	
				stream << "\tFold #" << (fold + 1)
						  << " Rep #" << (task / arguments.fold + 1) << " pred log like = "
						  << logLikelihood;
						  
                bool write = true;						  
						  
				if (write) logger->writeLine(stream);				  

				// Store value
				predLogLikelihood[task] = logLikelihood;				    				    				    				    				    				    
			};	
			
	// Run all tasks in parallel	
	if (nThreads > 1) {		
    	ccd.getLogger().setConcurrent(true);
    }
	scheduler.execute(oneTask);
	if (nThreads > 1) {
    	ccd.getLogger().setConcurrent(false);
     	ccd.getLogger().flush();		
     }
	
// 	auto end1 = std::chrono::steady_clock::now();	
// 	
// 	std::cerr << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count()	
// 			  << std::endl;
// 					
// #else // NEW_LOOP	
//     std::cerr << "OLD_LOOP" << std::endl;
// 	auto start2 = std::chrono::steady_clock::now();	
// 
// 	std::vector<real> weights;
// 
// 	selector.reseed();
// 	
// 	/* start code duplication */
// 	//std::vector<double> predLogLikelihood;
// 	for (int i = 0; i < arguments.foldToCompute; i++) {
// 		int fold = i % arguments.fold;
// 		if (fold == 0) {
// 			selector.permute(); // Permute every full cross-validation rep
// 		}
// 
// 		// Get this fold and update
// 		selector.getWeights(fold, weights);
// 		if(weightsExclude){
// 			for(int j = 0; j < (int)weightsExclude->size(); j++){
// 				if(weightsExclude->at(j) == 1.0){
// 					weights[j] = 0.0;
// 				}
// 			}
// 		}
// 		ccd.setWeights(&weights[0]);
// 		std::ostringstream stream;
// 		stream << "Running at " << ccd.getPriorInfo() << " ";
// 				
// 		if (coldStart) ccd.resetBeta();
// 		
// 		ccd.update(arguments.maxIterations, arguments.convergenceType, arguments.tolerance);
// 
// 		// Compute predictive loglikelihood for this fold
// 		selector.getComplement(weights); 
// 		if(weightsExclude){
// 			for(int j = 0; j < (int)weightsExclude->size(); j++){
// 				if(weightsExclude->at(j) == 1.0){
// 					weights[j] = 0.0;
// 				}
// 			}
// 		}
// 
// 		double logLikelihood = ccd.getPredictiveLogLikelihood(&weights[0]); 
// 
// 		stream << "Grid-point #" << (step + 1) << " at "; // << ccd.getHyperprior();
// 		std::vector<double> hyperprior = ccd.getHyperprior();
// 		std::copy(hyperprior.begin(), hyperprior.end(),
// 		    std::ostream_iterator<double>(stream, " "));
// 		
// 		stream << "\tFold #" << (fold + 1)
// 				  << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
// 				  << logLikelihood;
//         logger->writeLine(stream);				  
// 
// 		// Store value
//  		predLogLikelihood[i] = logLikelihood;
// 	}
// 	
// 	auto end2 = std::chrono::steady_clock::now();	
// 	
// 	std::cerr << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count()	
// 			  << std::endl;	
// #endif // NEW_LOOP			  

	double pointEstimate = computePointEstimate(predLogLikelihood);
	
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
	
	bool coldStart = arguments.resetCoefficients;
	
    // Start of new multi-thread set-up
	int nThreads = (arguments.threads == -1) ? 
	    std::thread::hardware_concurrency() :
	    arguments.threads;
	    
	std::ostringstream stream2;
	stream2 << "Using " << nThreads << " thread(s)";
	logger->writeLine(stream2);    	
	
	std::vector<CyclicCoordinateDescent*> ccdPool;
	std::vector<AbstractSelector*> selectorPool;
	
	ccdPool.push_back(&ccd);
	selectorPool.push_back(&selector);
		
	for (int i = 1; i < nThreads; ++i) {
		ccdPool.push_back(ccd.clone());
		selectorPool.push_back(selector.clone());
	}	
	// End of multi-thread set-up

	int step = 0;
	bool finished = false;
	
	while (!finished) {
		ccd.setHyperprior(tryvalue);
		selector.reseed();		

		std::vector<double> predLogLikelihood;

		// Newly re-located code
		double pointEstimate = doCrossValidation(ccd, selector, arguments, step, coldStart, 
			nThreads, ccdPool, selectorPool,
			predLogLikelihood);

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
	
	// Clean up
	for (int i = 1; i < nThreads; ++i) {
		delete ccdPool[i];
		delete selectorPool[i];		
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
