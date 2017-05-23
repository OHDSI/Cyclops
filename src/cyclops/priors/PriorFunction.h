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

namespace bsccs {
namespace priors {

class AbstractPriorFunction {
public:

    typedef std::pair<double,double> LocationScale;
    typedef std::vector<double> Arguments;

    AbstractPriorFunction(Arguments& arguments) : arguments(arguments) { }
    virtual ~AbstractPriorFunction() { }

    LocationScale operator()() const {
        return execute(arguments);
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

protected:
    virtual LocationScale execute(const Arguments& arguments) const;

private:
    const Arguments& arguments;
};

template <typename Func>
class PriorFunction : public AbstractPriorFunction {
public:
    PriorFunction(Func& function, Arguments& arguments) : AbstractPriorFunction(arguments) { }
    virtual ~PriorFunction() { }

protected:
    LocationScale execute(const Arguments& arguments) const {
        return function(arguments);
    }

private:
    const Func& function;
};

} /* namespace priors */
} /* namespace bsccs */
#endif /* PRIORFUNCTION_H_ */
