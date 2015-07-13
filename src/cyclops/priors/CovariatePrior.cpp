#include "priors/CovariatePrior.h"

namespace bsccs {
	namespace priors {

   PriorPtr CovariatePrior::makePrior(PriorType priorType, double variance) {
        PriorPtr prior;
        switch (priorType) {
            case NONE :
                prior = bsccs::make_shared<NoPrior>();
                break;
            case LAPLACE :
                prior = bsccs::make_shared<LaplacePrior>(variance);
                break;
            case NORMAL :
                prior = bsccs::make_shared<NormalPrior>(variance);
                break;
            default : break;
        }
        return prior;
    }
	} // namespace priors
} // namespace bsccs
