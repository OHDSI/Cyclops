/*
 * LeaveOneOutSelector.h
 *
 *  Created on: Mar 15, 2013
 *      Author: msuchard
 */

#ifndef LEAVEONEOUTSELECTOR_H_
#define LEAVEONEOUTSELECTOR_H_

#include "AbstractSelector.h"

namespace bsccs {

	class LeaveOneOutSelector : public AbstractSelector {
	public:
		LeaveOneOutSelector(
				const std::vector<int>& inIds,
				SelectorType inType,
				long inSeed);

		virtual ~LeaveOneOutSelector();

		void permute();
		void getWeights(int batch, std::vector<real>& weights);
		void getComplement(std::vector<real>& weights);
	private:
		std::vector<int> permutation;
		long currentID;

	};
} /* namespace bsccs */
#endif /* LEAVEONEOUTSELECTOR_H_ */
