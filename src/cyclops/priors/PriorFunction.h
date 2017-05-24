/*
 * PriorFunction.h
 *
 *  Created on: May 12, 2017
 *      Author: Marc A. Suchard
 *
 *  Describes API for prior with location/scale dependent on external parameters
 */

#ifndef PRIORFUNCTION_H_
#define PRIORFUNCTION_H_

#include "Types.h"

namespace bsccs {
namespace priors {

class PriorFunction {
public:

    typedef std::pair<double,double> LocationScale;
    typedef std::vector<double> Arguments;
    typedef bsccs::shared_ptr<Arguments> ArgumentsPtr;

    PriorFunction(ArgumentsPtr arguments) : arguments(arguments) { }
    virtual ~PriorFunction() { }

    LocationScale operator()() const {
        return execute(*arguments);
    }

    // LocationScale operator()(Arguments& arguments) const {
    //     if (arguments.size() != arity) {
    //         // TODO Throw error
    //     }
    //
    //     return execute(arguments);
    // }
    //
    // unsigned int getArity() const { return arity; }

    Arguments& getArguments() { return *arguments; }

protected:
    virtual LocationScale execute(const Arguments& arguments) const = 0;

private:
    const ArgumentsPtr arguments;
};

// typedef bsccs::shared_ptr<PriorFunction> PriorFunctionPtr;
typedef std::unique_ptr<PriorFunction> PriorFunctionPtr;
typedef bsccs::shared_ptr<std::vector<double>> PriorFunctionParameterPtr;

// template <typename Func>
// class PriorFunction : public AbstractPriorFunction {
// public:
//     PriorFunction(Func& function, Arguments& arguments) : AbstractPriorFunction(arguments) { }
//     virtual ~PriorFunction() { }
//
// protected:
//     LocationScale execute(const Arguments& arguments) const {
//         return function(arguments);
//     }
//
// private:
//     const Func& function;
// };

} /* namespace priors */
} /* namespace bsccs */
#endif /* PRIORFUNCTION_H_ */
