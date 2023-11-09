/*
 * WeightBasedSelector.h
 *
 *  Created on: Aug 26, 2022
 *      Author: msuchard
 */

#ifndef WEIGHTBASED_H_
#define WEIGHTBASED_H_

// #include <vector>

#include "CrossValidationSelector.h"

namespace bsccs {

class WeightBasedSelector : public AbstractSelector {
public:
	WeightBasedSelector(
			int inFold,
			std::vector<int> inIds,
			SelectorType inType,
			long inSeed,
    	    loggers::ProgressLoggerPtr _logger,
	    	loggers::ErrorHandlerPtr _error,
			std::vector<double>* wtsExclude,
			std::vector<double>* wtsOriginal);

	virtual ~WeightBasedSelector();

	void permute();

 	void reseed();

	void getWeights(int batch, std::vector<double>& weights);

	void getComplement(std::vector<double>& weights);

	AbstractSelector* clone() const;
// 
private:
// 	int fold;
// 	std::vector<int> permutation;
// 	std::vector<int> intervalStart;
	std::vector<double>* weightsExclude;
	std::vector<double>* weightsOriginal;
};

} // namespace

#endif /* WEIGHTBASED_H_ */
