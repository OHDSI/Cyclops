/*
 * CrossValidation.cpp
 *
 *  Created on: Sep 9, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <algorithm>
#include <iterator>
#include <set>

#include "CrossValidationSelector.h"

namespace bsccs {

using std::vector;
using std::set;
using std::insert_iterator;

CrossValidationSelector::CrossValidationSelector(
		int inFold,
		std::vector<int> inIds,
		SelectorType inType,
		long inSeed,
	    loggers::ProgressLoggerPtr _logger,
		loggers::ErrorHandlerPtr _error,
		std::vector<double>* wtsExclude,
		std::vector<double>* wtsOriginal) : AbstractSelector(inIds, inType, inSeed, _logger, _error), fold(inFold) {

	// Calculate interval starts
	intervalStart.reserve(fold + 1);
	int index = 0;
	int fraction = N / fold;
	int extra = N - fraction * fold;
	for (int i = 0; i < fold; i++) {
		intervalStart.push_back(index);
		index += fraction;
		if (i < extra) {
			index++;
		}
	}
	intervalStart.push_back(N);

    std::ostringstream stream;
	stream << "Performing " << fold << "-fold cross-validation [seed = "
		      << seed << "] with data partitions of sizes";

	for (int i = 0; i < fold; ++i) {
		stream << " " << (intervalStart[i+1] - intervalStart[i]);
	}
	logger->writeLine(stream);

	permutation.resize(N);

	weightsExclude = wtsExclude;
	weightsOriginal = wtsOriginal;
}

void CrossValidationSelector::reseed() {
//	std::cerr << "RESEEDING"  << std::endl;
	prng.seed(seed);
	for (size_t i = 0; i < N; ++i) {
		permutation[i] = i;
	}
}

void CrossValidationSelector::setFixedFolds(const std::vector<int>& foldOfRow) {
	if (foldOfRow.size() != K) {
		std::ostringstream stream;
		stream << "Need a cross-validation fold for each of the " << K << " rows";
		error->throwError(stream);
	}
	// Fold of each exchangeable unit: a row, or the stratum or subject that ids gives. A row without a fold (-1: weight
	// 0) takes its unit's, or fold 0 in a unit of such rows only
	std::vector<int> foldOfUnit(N, -1);
	for (size_t k = 0; k < K; ++k) {
		const int f = foldOfRow[k];
		if (f == -1) {
			continue;
		}
		if (f < 0 || f >= fold) {
			std::ostringstream stream;
			stream << "Cross-validation fold " << f << " of row " << k << " is outside 0 to " << (fold - 1);
			error->throwError(stream);
		}
		const size_t unit = (type == SelectorType::BY_PID) ? ids[k] : k;
		if (foldOfUnit[unit] == -1) {
			foldOfUnit[unit] = f;
		} else if (foldOfUnit[unit] != f) {
			std::ostringstream stream;
			stream << "The rows of one stratum or subject must be in the same cross-validation fold";
			error->throwError(stream);
		}
	}
	// Units in fold order, so that fold f holds out permutation[intervalStart[f], intervalStart[f + 1]); a fold must hold
	// out a unit with a fold given
	std::vector<int> given(fold, 0), count(fold, 0);
	for (size_t unit = 0; unit < N; ++unit) {
		if (foldOfUnit[unit] != -1) {
			++given[foldOfUnit[unit]];
		} else { // rows of weight 0 only (ids are 0, ..., N - 1: see AbstractSelector)
			foldOfUnit[unit] = 0;
		}
		++count[foldOfUnit[unit]];
	}
	intervalStart.assign(1, 0);
	for (int f = 0; f < fold; ++f) {
		if (given[f] == 0) {
			std::ostringstream stream;
			stream << "Cross-validation fold " << f << " holds out no rows";
			error->throwError(stream);
		}
		intervalStart.push_back(intervalStart.back() + count[f]);
	}
	fixedPermutation.resize(N);
	std::vector<int> next(intervalStart.begin(), intervalStart.end() - 1);
	for (size_t unit = 0; unit < N; ++unit) {
		fixedPermutation[next[foldOfUnit[unit]]++] = static_cast<int>(unit);
	}

	std::ostringstream stream;
	stream << "Using the given " << fold << " cross-validation folds of sizes";
	for (int f = 0; f < fold; ++f) {
		stream << " " << given[f];
	}
	logger->writeLine(stream);
}

bool CrossValidationSelector::getFoldOfRow(std::vector<int>& foldOfRow) const {
	std::vector<int> foldOfUnit(N, -1);
	for (int f = 0; f < fold; ++f) {
		for (int i = intervalStart[f]; i < intervalStart[f + 1]; ++i) {
			foldOfUnit[permutation[i]] = f;
		}
	}
	foldOfRow.resize(K);
	for (size_t k = 0; k < K; ++k) {
		foldOfRow[k] = foldOfUnit[(type == SelectorType::BY_PID) ? ids[k] : k];
	}
	return true;
}

CrossValidationSelector::~CrossValidationSelector() {
	// Do nothing
}

void CrossValidationSelector::getWeights(int batch, std::vector<double>& weights) {
	if (weights.size() != K) {
		weights.resize(K);
	}

	if (weightsOriginal != nullptr) {
	    std::copy(weightsOriginal->begin(), weightsOriginal->end(), weights.begin());
	} else {
	    std::fill(weights.begin(), weights.end(), 1.0);
	}

	if (batch == -1) {
		return;
	}

	if (type == SelectorType::BY_PID) {
		std::set<int> excludeSet;
		std::copy(
				permutation.begin() + intervalStart[batch],
				permutation.begin() + intervalStart[batch + 1],
				insert_iterator< std::set<int> >(excludeSet, excludeSet.begin())
				);

		for (size_t k = 0; k < K; k++) {
			if (excludeSet.find(ids.at(k)) != excludeSet.end()) { // found
				weights[k] = 0.0;
			} else {
			    if (weightsOriginal != nullptr) {
			        weights[k] = (*weightsOriginal)[k];
			    } else {
				    weights[k] = 1.0;
			    }
			}
		}
	} else { // SelectorType::BY_ROW
// 		std::fill(weights.begin(), weights.end(), 0.0);
// 		std::fill(weights.begin(), weights.begin() + 100, 1.0);
		std::for_each(
			permutation.begin() + intervalStart[batch],
			permutation.begin() + intervalStart[batch + 1],
			[&weights](const int excludeIndex) {
				weights[excludeIndex] = 0.0;
		});
	}
}

AbstractSelector* CrossValidationSelector::clone() const {
	return new (std::nothrow) CrossValidationSelector(*this); // default copy constructor
}

void CrossValidationSelector::getComplement(std::vector<double>& weights) {
    if (weightsOriginal != nullptr) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weights[i] == 0.0) {
                weights[i] = (*weightsOriginal)[i];
            } else {
                weights[i] = 0.0;
            }
        }
    } else {
	    for (auto it = weights.begin(); it != weights.end(); it++) {
		    *it = 1 - *it;
	    }
    }
}

void CrossValidationSelector::permute() {

	if (!fixedPermutation.empty()) { // the given folds, in every repetition
		permutation = fixedPermutation;
		return;
	}

	// Do random shuffle
	if (!deterministic) {
//      	std::cerr << "PERMUTE" << std::endl;
		std::shuffle(permutation.begin(), permutation.end(), prng);
	}

	if(weightsExclude){
		vector<int> permutationCopy = permutation;
		int nExcluded = 0;
		for(int i = 0; i < (int)weightsExclude->size(); i++){
			if(weightsExclude->at(i) != 0.0){
				nExcluded++;
			}
		}
		int fraction = nExcluded / fold;
		int extra = nExcluded - fraction * fold;

		vector<int> nExcludedPerFold;
		for(int i = 0; i < fold; i++){
			if(i < extra){
				nExcludedPerFold.push_back(fraction + 1);
			}
			else{
				nExcludedPerFold.push_back(fraction);
			}
		}
		int foldIncluded = 0;
		int foldExcluded = 0;
		int nextExcluded = intervalStart[0];
		int nextIncluded = intervalStart[0] + nExcludedPerFold[0];
		for(size_t i = 0; i < permutationCopy.size(); i++){
			if(weightsExclude->at(permutationCopy[i]) == 0.0){
				permutation[nextIncluded] = permutationCopy[i];
				nextIncluded++;
				if(nextIncluded == intervalStart[foldIncluded + 1]){
					nextIncluded = intervalStart[foldIncluded + 1] + nExcludedPerFold[foldIncluded + 1];
					foldIncluded++;
				}
			}
			else{
				permutation[nextExcluded] = permutationCopy[i];
				nextExcluded++;
				if(nextExcluded == intervalStart[foldExcluded] + nExcludedPerFold[foldExcluded]){
					nextExcluded = intervalStart[foldExcluded + 1];
					foldExcluded++;
				}
			}
		}
	}
}

} // namespace
