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

#include "AutoSearchCrossValidationDriver.h"
#include "CyclicCoordinateDescent.h"
#include "CrossValidationSelector.h"
#include "AbstractSelector.h"
#include "ccd.h"
#include "../utils/HParSearch.h"

namespace bsccs {

AutoSearchCrossValidationDriver::AutoSearchCrossValidationDriver(
			const ModelData& _modelData,
			int iGridSize,
			double iLowerLimit,
			double iUpperLimit,
			vector<real>* wtsExclude) : modelData(_modelData), maxPoint(0), gridSize(iGridSize),
			lowerLimit(iLowerLimit), upperLimit(iUpperLimit), weightsExclude(wtsExclude) {

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
		cerr << "Unable to open log file: " << arguments.cvFileName << endl;
		exit(-1);
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

void AutoSearchCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Check that selector is type of CrossValidationSelector

	std::vector<real> weights;

	double tryvalue = modelData.getNormalBasedDefaultVar();
	UniModalSearch searcher;
	const double eps = 0.05; //search stopper

	std::cout << "Default var = " << tryvalue << std::endl;

//	using std::cout;
//	using std::map;
//    map<double,double> y_by_x;
//    //example llkl.d001145.b prepared in R
//    y_by_x[ 2.000000e+04 ] = -408.199;
//    y_by_x[ 2.002884e+03 ] = -328.581;
//    y_by_x[ 2.000000e+02 ] = -281.788;
//    y_by_x[ 2.002884e+01 ] = -229.966;
//    y_by_x[ 2.000000e+00 ] = -184.908;
//    y_by_x[ 2.002884e-01 ] = -150.792;
//    y_by_x[ 2.000000e-02 ] = -132.081;
//    y_by_x[ 2.002884e-03 ] = -127.708;
//    y_by_x[ 2.000000e-04 ] = -146.397;
//    y_by_x[ 2.002884e-05 ] = -183.208;
//    QuadrCoefs c = QuadrLogFit( y_by_x );
//    //R results:
//    // (Intercept)     logvar2      logvar
//    // -171.030466   -1.186039  -12.662184
//    cout<<"c0="<<c.c0<<"  c1="<<c.c1<<"  c2="<<c.c2<<endl;
//
//    UniModalSearch s;
//    for( map<double,double>::const_iterator itr=y_by_x.begin(); itr!=y_by_x.end(); itr++ )
//        s.tried( itr->first, itr->second );
//    //R result: 0.004805411
//    cout<<"step="<<s.step().second<<endl;
//    exit(-1);


	bool finished = false;

	int step = 0;
	while (!finished) {
		ccd.setHyperprior(tryvalue);

		/* start code duplication */
		std::vector<double> predLogLikelihood;
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
			std::cout << "Running at " << ccd.getPriorInfo() << " ";
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

			std::cout << "Grid-point #" << (step + 1) << " at " << tryvalue;
			std::cout << "\tFold #" << (fold + 1)
			          << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
			          << logLikelihood << std::endl;

			// Store value
			predLogLikelihood.push_back(logLikelihood);
		}

		double pointEstimate = computePointEstimate(predLogLikelihood);
		/* end code duplication */

		double stdDevEstimate = computeStDev(predLogLikelihood, pointEstimate);

		std::cout << "AvgPred = " << pointEstimate << " with stdev = " << stdDevEstimate << std::endl;
        searcher.tried(tryvalue, pointEstimate, stdDevEstimate);
        pair<bool,double> next = searcher.step();
        std::cout << "Completed at " << tryvalue << std::endl;
        std::cout << "Next point at " << next.second << " and " << next.first << std::endl;

        tryvalue = next.second;
        if (!next.first) {
            finished = true;
        }
        std::cout << searcher;
        step++;
        if (step >= 10) exit(-1);
	}

	maxPoint = tryvalue;

	// Report results

	std::cout << std::endl;
	std::cout << "Maximum predicted log likelihood estimated at:" << std::endl;
	std::cout << "\t" << maxPoint << " (variance)" << std::endl;
	if (!arguments.useNormalPrior) {
		double lambda = convertVarianceToHyperparameter(maxPoint);
		std::cout << "\t" << lambda << " (lambda)" << std::endl;
	}
	std:cout << std::endl;
	
/*  Code taken from BBR ZO.cpp
        //prepare CV
        const double randmax = RAND_MAX;
        unsigned nfolds = hyperParamPlan.nfolds(); //10
        if( nfolds>drs.n() ) {
            nfolds = drs.n();
            Log(1)<<"\nWARNING: more folds requested than there are data. Reduced to "<<nfolds;
        }
        unsigned nruns = hyperParamPlan.nruns();
        if( nruns>nfolds )  nruns = nfolds;
        vector<int> rndind = PlanStratifiedSplit( drs, nfolds );

        //prepare search
        double tryvalue = NormBasedDefaultVar( drs.dim(), stats, m_modelType.Standardize() );
        UniModalSearch searcher;
        const double eps = 0.05; //search stopper

        Vector localbeta = priorMode; //reuse this through all loops; can do better?

        //search loop
        while( true )
        {
            //try tryvalue
            BayesParameter bp = hyperParamPlan.InstByVar( tryvalue );
            double lglkli = 0; //would be the result for tryvalue
            double lglkliSqu = 0; //for stddev

            vector<bool> foldsAlreadyRun( nfolds, false );
            for( unsigned irun=0; irun<nruns; irun++ ) //cv loop
            {
                //next fold to run
                unsigned x = unsigned( rand()*(nfolds-irun)/(randmax+1) );
                unsigned ifold, y=0;
                for( ifold=0; ifold<nfolds; ifold++ )
                    if( !foldsAlreadyRun[ifold] ) {
                        if( y==x ) break;
                        else y++;
                    }
                foldsAlreadyRun[ifold] = true;
                Log(5)<<"\nCross-validation "<<nfolds<<"-fold; Run "<<irun+1<<" out of "<<nruns<<", fold "<<ifold+1;

                //run cv part for a given fold
                SamplingRowSet cvtrain( drs, rndind, ifold, ifold+1, false ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
                SamplingRowSet cvtest( drs, rndind, ifold, ifold+1, true ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
                Log(5)<<"\ncv training sample "<<cvtrain.n()<<" rows"<<"  test sample "<<cvtest.n()<<" rows";

                // build the model
                InvData invData( cvtrain );
                localbeta = ZOLR( invData,
                    bp, priorMode, priorScale, priorSkew, 
                    m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
                    localbeta );
                    //Log(9)<<"\nlocalBeta "<<localbeta;

                double run_lglkli = LogLikelihood( cvtest, m_modelType, localbeta );
                lglkli = lglkli*irun/(irun+1) + run_lglkli/(irun+1); //average across runs
                lglkliSqu = lglkliSqu*irun/(irun+1) + run_lglkli*run_lglkli/(irun+1); //for stddev

            }//cv loop

            double lglkli_stddev = sqrt( lglkliSqu - lglkli*lglkli );
            Log(5)<<"\nPrior variance "<<tryvalue<<" \tcv loglikeli "<<lglkli<<" \tstddvev "<<lglkli_stddev;
            searcher.tried( tryvalue, lglkli, lglkli_stddev );
            pair<bool,double> step = searcher.step();
            tryvalue = step.second;
            if( ! step.first )
                break;                      //----->>----
        }//search loop
        Log(3)<<"\nBest prior variance "<<tryvalue; 

        //build final model
        m_bayesParam = hyperParamPlan.InstByVar( tryvalue ); //SIC! use smoothing from the model //was searcher.bestx();
        Log(3)<<std::endl<<"Starting final model, Time "<<Log.time();

    }  // new: search without a grid	
*/	
}


//void AutoSearchCrossValidationDriver::findMax(double* maxPoint, double* maxValue) {
//
//	*maxPoint = gridPoint[0];
//	*maxValue = gridValue[0];
//	for (int i = 1; i < gridPoint.size(); i++) {
//		if (gridValue[i] > *maxValue) {
//			*maxPoint = gridPoint[i];
//			*maxValue = gridValue[i];
//		}
//	}
//}

} // namespace
