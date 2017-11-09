/*
 * CrossValidation.h
 *
 *  Created on: Sep 9, 2010
 *      Author: msuchard
 */

#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include <vector>

#include "AbstractSelector.h"

namespace bsccs {

class CrossValidationSelector : public AbstractSelector {
public:
	CrossValidationSelector(
			int inFold,
			std::vector<int> inIds,
			SelectorType inType,
			long inSeed,
    	    loggers::ProgressLoggerPtr _logger,
	    	loggers::ErrorHandlerPtr _error,
			std::vector<double>* wtsExclude,
			std::vector<double>* wtsOriginal);

	virtual ~CrossValidationSelector();

	void permute();

 	void reseed();

	void getWeights(int batch, std::vector<double>& weights);

	void getComplement(std::vector<double>& weights);

	AbstractSelector* clone() const;

private:
	int fold;
	std::vector<int> permutation;
	std::vector<int> intervalStart;
	std::vector<double>* weightsExclude;
	std::vector<double>* weightsOriginal;
};

} // namespace

#endif /* CROSSVALIDATION_H_ */
