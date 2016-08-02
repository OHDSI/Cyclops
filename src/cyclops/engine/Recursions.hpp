/*
 * Recursions.hpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Marc A. Suchard
 *      Author: Yuxi Tian
 */

#ifndef RECURSIONS_HPP_
#define RECURSIONS_HPP_

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>

namespace bsccs {

class bigNum {
public:
	int exp;
	double frac;
	union DoubleInt {
		unsigned i[2];
		double d;
	};
	bigNum (double a) {
		frac = frexp(a,&exp);
	}
	bigNum (double f, int e) {
		frac = f;
		exp = e;
	}
	static bigNum mul (bigNum a, bigNum b) {
		if (a.isZero() || b.isZero()) {
			return bigNum(0);
		}
		int e = 0;
		double f = a.frac * b.frac;
		if (f < 0.5) {
			e = -1;
			f = 2*f;
		}
		e = e + a.exp + b.exp;
		return bigNum(f,e);
	}
	static bigNum mul (double a, bigNum b) {
		if (a==0 || b.isZero()) {
			return bigNum(0);
		}
		int e;
		double f = frexp(a * b.frac, &e);
		e = e + b.exp;
		return bigNum(f,e);
	}
	static bigNum div (bigNum a, bigNum b) {
		if (a.isZero()) {
			return bigNum(0);
		}
		int e = 0;
		double f = a.frac/b.frac;
		if (f >= 1) {
			e = 1;
			f = f/2;
		}
		e = e + a.exp - b.exp;
		return bigNum(f,e);
	}
/*
	static bigNum add (bigNum a, bigNum b) {
		int n = b.exp - a.exp;
		if (n > 100) {
			return b;
		}
		if (n < -100) {
			return a;
		}
		int e;
		double f = frexp(a.frac * pow(2.0, static_cast<double>(a.exp-b.exp)) + b.frac, &e);
		e = e + b.exp;

		printf("%f \n", bigNum(f,e).toDouble());
		printf("%f \n", bigNum::add1(a,b).toDouble());
		printf("%f %d %f %d \n", a.frac, a.exp, b.frac, b.exp);

		return bigNum(f,e);
	}
*/

	static bigNum add(bigNum a, bigNum b) {
		int n = b.exp - a.exp;
		if (n > 100 || a.isZero()) {
			return b;
		}
		if (n < -100 || b.isZero()) {
			return a;
		}
		DoubleInt x;
		x.d = b.frac;
		x.i[1] = (((x.i[1]>>20)+n)<<20)|((x.i[1]<<12)>>12);
		int e;
		double f = frexp(x.d + a.frac, &e);
		e = e + a.exp;
		return bigNum(f,e);
	}

	bool isZero() {
		return (frac==0 && exp==0);
	}


	double toDouble() const {
		return frac * std::pow(2.0, static_cast<double>(exp)); // TODO Use shift operator
	}
};
    
namespace sugar {        
    // Syntactic sugar to write: bigNum + bigNum
    template <typename T>
    T operator+(const T& lhs, const T& rhs) {
    	return T::add(lhs, rhs);    
    } 
   
	// Syntactic sugar to write: bigNum / bigNum
	template <typename T>
	T operator/(const T& lhs, const T rhs) {
		return T::div(lhs, rhs);
	}
	
	// Syntactic sugar to write: double * bigNum
	template <typename T>
	T operator*(const double lhs, const T& rhs) {
		return T::mul(lhs, rhs);
	}
	
	// Syntactic sugar to write: bigNum * bigNum
	template <typename T>
	T operator*(const T& lhs, const T& rhs) {
		return T::mul(lhs, rhs);
	}	
	
	// Syntactic sugar to write: bigNum - bigNum
	template <typename T>
	double operator-(const T& lhs, const T& rhs) {
		return lhs.toDouble() - rhs.toDouble();
	}
	
	// Syntactic sugar to write: bigNum += bigNum // TODO Is this needed?
	
} // namespace sugar


template <typename T, typename UIteratorType, typename W, typename SparseIteratorType>
std::vector<T> computeHowardRecursion(UIteratorType itExpXBeta, SparseIteratorType itX, 
	int numSubjects, int numCases, /*bsccs::real*  */ W caseOrNo) {

	using namespace sugar;

	double caseSum = 0;

	std::vector<T> B[2];
	std::vector<T> dB[2];
	std::vector<T> ddB[2];

	int currentB = 0;

	B[0].push_back(1);
	B[1].push_back(1);
	dB[0].push_back(0);
	dB[1].push_back(0);
	ddB[0].push_back(0);
	ddB[1].push_back(0);

	for (int i=1; i<=numCases; i++) {
		B[0].push_back(0);
		B[1].push_back(0);
		dB[0].push_back(0);
		dB[1].push_back(0);
		ddB[0].push_back(0);
		ddB[1].push_back(0);
	}
	
	//double maxXi = 0.0;
	//double maxSorted = 0.0;
	//std::vector<T> sortXi;
	
	for (int n=1; n<= numSubjects; n++) {
		for (int m=std::max(1,n+numCases-numSubjects); m<=std::min(n,numCases);m++) {
			  B[!currentB][m] =   B[currentB][m] + (*itExpXBeta) *   B[currentB][m-1];
			 dB[!currentB][m] =  dB[currentB][m] + (*itExpXBeta) *  dB[currentB][m-1] + (*itX)*(*itExpXBeta)*B[currentB][m-1];
			ddB[!currentB][m] = ddB[currentB][m] + (*itExpXBeta) * ddB[currentB][m-1] + (*itX)*(*itX)*(*itExpXBeta)*B[currentB][m-1] + 
								2*(*itX)*(*itExpXBeta)*dB[currentB][m-1];
		}
		if (caseOrNo[n-1] == 1) {
			caseSum += (*itX);
		}

		//if (*itX > maxXi) {
		//	maxXi = *itX;
		//}
		//sortXi.push_back(*itX);

		currentB = !currentB;
		++itExpXBeta; ++itX;

	}

	//std::sort (sortXi.begin(), sortXi.end());
	//for (int i=1; i<=numCases; i++) {
	//	maxSorted += sortXi[numSubjects-i];
	//}
	//maxXi = maxXi * numCases;

	std::vector<T> result;
	result.push_back(B[currentB][numCases]);
	result.push_back(dB[currentB][numCases]);
	result.push_back(ddB[currentB][numCases]);
	result.push_back(caseSum);
	
	//result.push_back(maxXi);
	//result.push_back(maxSorted);

	return result;
}
            
} // namespace

#endif /* RECURSIONS_HPP_ */
