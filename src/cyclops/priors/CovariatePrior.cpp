#include "priors/CovariatePrior.h"

namespace bsccs {
	namespace priors {
		
   PriorPtr CovariatePrior::makePrior(PriorType priorType) {
        PriorPtr prior;
        switch (priorType) {
            case NONE :
                prior = bsccs::make_shared<NoPrior>();
                break;
            case LAPLACE :
                prior = bsccs::make_shared<LaplacePrior>();
                break;
            case NORMAL :
                prior = bsccs::make_shared<NormalPrior>();
                break;
            default : break;    
        }
        return prior;
    }
	} // namespace priors
} // namespace bsccs