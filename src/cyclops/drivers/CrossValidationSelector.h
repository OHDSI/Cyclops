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

	// Folds given by the user (0, 1, ..., inFold - 1 for each row in data order) instead of random ones; the rows
	// of one exchangeable unit (stratum or subject) must share a fold, and every fold must hold out a unit
	void setFixedFolds(const std::vector<int>& foldOfRow);

	bool getFoldOfRow(std::vector<int>& foldOfRow) const;

private:
	int fold;
	std::vector<int> permutation;
	std::vector<int> intervalStart;
	std::vector<int> fixedPermutation; // units ordered by their given fold; empty for random folds
	std::vector<double>* weightsExclude;
	std::vector<double>* weightsOriginal;
};

} // namespace

#endif /* CROSSVALIDATION_H_ */
