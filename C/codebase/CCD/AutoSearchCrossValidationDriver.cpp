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
			vector<real>* wtsExclude) : modelData(_modelData), gridSize(iGridSize),
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

	string sep(","); // TODO Make option

	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

	for (int i = 0; i < gridPoint.size(); i++) {
		outLog << std::setw(5) << std::setprecision(4) << std::fixed << gridPoint[i] << sep;
		if (!arguments.useNormalPrior) {
			outLog << convertVarianceToHyperparameter(gridPoint[i]) << sep;
		}
		outLog << std::scientific << gridValue[i] << sep;
		outLog << (maxValue - gridValue[i]) << std::endl;
	}

	outLog.close();
}

void AutoSearchCrossValidationDriver::resetForOptimal(
		CyclicCoordinateDescent& ccd,
		CrossValidationSelector& selector,
		const CCDArguments& arguments) {

	ccd.setWeights(NULL);

	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);
	ccd.setHyperprior(maxPoint);
	ccd.resetBeta(); // Cold-start
}


/*double AvgSquNorm( IRowSet& rs ) //for default bayes parameter: avg x*x
{
    rs.rewind();
    double avgss = 0;
    double n = 0;
    while( rs.next() ) {
        SparseVector x = rs.xsparse();
        double xsqu = 0;
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            xsqu += ix->second * ix->second;
        avgss = avgss*n/(n+1) + xsqu/(n+1);
        n ++;
    }
    return avgss;
}*/

//static double AutoSearchCrossValidationDriver::avgSquNorm() {
//	CyclicCoordinateDescent& ccd;
//	ccd.
//}

// Taken from BBR
double AutoSearchCrossValidationDriver::normBasedDefaultVar() {

	return
//    double avgsqunorm = standardize ? 1.0 : AvgSquNorm(stats);
//    double priorVar = dim/avgsqunorm;
//    Log(3)<<"\nAvg square norm (no const term) "<<avgsqunorm<<" Prior var "<<priorVar;
//    return priorVar;
}


void AutoSearchCrossValidationDriver::drive(
		CyclicCoordinateDescent& ccd,
		AbstractSelector& selector,
		const CCDArguments& arguments) {

	// TODO Check that selector is type of CrossValidationSelector

	std::vector<real> weights;

	double tryvalue = 1.0; /*NormBasedDefaultVar( drs.dim(), stats, m_modelType.Standardize() );*/
	UniModalSearch searcher;
	const double eps = 0.05; //search stopper

	bool finished = false;

	int step = 0;
	while (!finished) {

		std::vector<double> predLogLikelihood;
		double point = computeGridPoint(step);
		ccd.setHyperprior(point);

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

			std::cout << "Grid-point #" << (step + 1) << " at " << point;
			std::cout << "\tFold #" << (fold + 1)
			          << " Rep #" << (i / arguments.fold + 1) << " pred log like = "
			          << logLikelihood << std::endl;

			// Store value
			predLogLikelihood.push_back(logLikelihood);
		}

		double value = computePointEstimate(predLogLikelihood) /
				(double(arguments.foldToCompute) / double(arguments.fold));
		gridPoint.push_back(point);
		gridValue.push_back(value);
	}

	// Report results
	double maxPoint;
	double maxValue;
	findMax(&maxPoint, &maxValue);

	std::cout << std::endl;
	std::cout << "Maximum predicted log likelihood (" << maxValue << ") found at:" << std::endl;
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


void AutoSearchCrossValidationDriver::findMax(double* maxPoint, double* maxValue) {

	*maxPoint = gridPoint[0];
	*maxValue = gridValue[0];
	for (int i = 1; i < gridPoint.size(); i++) {
		if (gridValue[i] > *maxValue) {
			*maxPoint = gridPoint[i];
			*maxValue = gridValue[i];
		}
	}
}

} // namespace
