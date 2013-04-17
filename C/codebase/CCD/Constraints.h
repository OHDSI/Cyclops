/*
 * Constraints.h
 *
 *  Created on: Mar 16, 2013
 *      Author: msuchard
 */

#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

namespace bsccs {

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif

//template <class Derived>
//struct Constraint : Derived {
//
//	bool hasConstraint(int index) {
//		return static_cast<Derived*>(this)->hasConstraint(index);
//	}
//};
//
//struct NoConstraint : Constraint<NoConstraint> {
//	bool hasContraint(int index) {
//		return false;
//	}
//};

struct Constraint {
	virtual ~Constraint() { /* Do nothing */ }
	virtual real getConstrainedDelta(real beta, real delta) = 0; // Pure virtual
};

struct NoConstraint : public Constraint {
	real getConstrainedDelta(real beta, real delta) {
		return delta;
	}
};

struct NonNegativityConstraint : public Constraint {
	real getConstrainedDelta(real beta, real delta) {
		real minusBeta = -beta;
		return (minusBeta > delta) ? minusBeta : delta;
	}
};

} // namespace

#endif /* CONSTRAINTS_H_ */
