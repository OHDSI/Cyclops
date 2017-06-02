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

class PriorFunction : public CacheCallback {
public:

    typedef std::pair<double,double> LocationScale;
    typedef std::vector<double> Arguments;
    typedef bsccs::shared_ptr<Arguments> ArgumentsPtr;
    typedef std::vector<double> Evaluation;
    typedef std::vector<Evaluation> ResultSet;


    PriorFunction(const std::vector<double>& startingParameters) : CacheCallback()//,
           // variancePtrs(variancePtrs)
        {
        for (unsigned int i = 0; i < startingParameters.size(); ++i) {
            VariancePtr ptr(bsccs::make_shared<double>(startingParameters[i]), this);
            variancePtrs.emplace_back(ptr);
           // variancePtrs[i]->setCallback(this);
        }

    }

    virtual ~PriorFunction() { }

    // LocationScale operator()() const {
    //     return execute(*arguments);
    // }

    const unsigned int getMaxIndex() {
        if (!isValid()) {
            makeValid();
        }
        const auto length = results.size();
        // std::cerr << "maxindex = " << length << std::endl;
        return length;
    }

    const Evaluation& operator()(unsigned int index) {
        // std::cerr << "operator()(usigned int)" << std::endl;
        if (!isValid()) {
            makeValid();
        }
        return results[index];
    }

    std::vector<VariancePtr> getVarianceParameters() const {
        return variancePtrs;
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

    // Arguments& getArguments() { return *arguments; }

protected:
    virtual ResultSet execute(const Arguments& arguments) const = 0;

private:
    void makeValid() {
        // std::cerr << "makeValid()" << std::endl;

        Arguments arguments;
        for (unsigned int i = 0; i < variancePtrs.size(); ++i) {
            // std::cerr << "try const" << std::endl;
            arguments.push_back(variancePtrs[i].get());
            // std::cerr << "end try" << std::endl;
        }
        results = execute(arguments);
        setValid(true);
    }

   // const ArgumentsPtr arguments;
    std::vector<VariancePtr> variancePtrs;
    ResultSet results;
};

// typedef bsccs::shared_ptr<PriorFunction> PriorFunctionPtr;
// typedef std::unique_ptr<PriorFunction> PriorFunctionPtr;
typedef bsccs::shared_ptr<PriorFunction> PriorFunctionPtr;
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
