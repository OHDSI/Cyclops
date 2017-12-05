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


template <typename T, typename SparseIteratorType, typename UIteratorType>//, typename thread_pool> //typename W typename UIteratorType,
std::vector<T> computeHowardRecursion(UIteratorType itExpXBeta, SparseIteratorType itX,
	int numSubjects, int numCases){//, thread_pool &threadPool){//, /*bsccs::real*  */ W caseOrNo) {

    //std::cout<<"starting recursion\n";
	using namespace sugar;
	std::vector<T> result;
	//double caseSum = 0;

	if (numCases==1) {
		T B = 0;
		T dB = 0;
		T ddB = 0;
		for (int n=0; n<numSubjects; n++) {
			B += *itExpXBeta;
			dB += *itExpXBeta * *itX;
			ddB += *itExpXBeta * *itX * *itX;
			++itExpXBeta;
			++itX;
		}
		result.push_back(B);
		result.push_back(dB);
		result.push_back(ddB);
	} else {

	    //std::vector<T> B[2];
	    // T B[2][3*(numCases+1)];
	    // B[0][0] = 1;
	    // B[1][0] = 1;
	    // for (int i=1; i<=3*numCases+2; i++) {
	    //     B[0][i] = 0;
	    //     B[1][i] = 0;
	    // }
	 	std::vector<T> B[2];
	//     B.emplace_back(std::vector<T>(1, static_cast<T>(1)));
	//     B.emplace_back(std::vector<T>(1, static_cast<T>(1)));

		// std::vector<T> B1;
		int currentB = 0;
		// int nThreads = 4;

		B[0].push_back(1);
		B[1].push_back(1);
		for (int i=1; i<=3*numCases+2; i++) {
			B[0].push_back(0);
			B[1].push_back(0);
		}

		int start = 1;
		int end = 0;
		// tbb::task_group threadPool;

		//std::future<void> futures[4];

		for (int n=1; n<= numSubjects; n++) {
			T x = *itX;
			T t = *itExpXBeta;
			if (n>numSubjects-numCases+1) start++;
			if (n<=numCases) end++;
			// int nloop = end-start+1;
			//std::cout<<"before"<<B[!currentB][3*start]<<'\n';
			//std::future<void> futures[4];
/*
			for (int i=0; i<nThreads; ++i) {
			    std::future<void> tempFuture = threadPool.push([x,t,currentB,&B](int id, int tStart,int tEnd){
			        //futures.push_back(std::move(threadPool.enqueue([=,&B](int tStart,int tEnd){
			        for (int m=tStart; m<tEnd; m++) {
			            T b = B[currentB][3*m-3];
			            T db = B[currentB][3*m-2];
			            T tb = t*b;
			            T xtb = x*tb;
			            T tdb = t*db;
			            B[!currentB][3*m] = B[currentB][3*m] + tb;
			            B[!currentB][3*m+1] = B[currentB][3*m+1] + tdb + xtb;
			            B[!currentB][3*m+2] = B[currentB][3*m+2] + t * B[currentB][3*m-1] + x*xtb + 2*x*tdb;
			        }
			    },start+i*nloop/nThreads,start+((i+1)==nThreads?nloop:(t+1)*nloop/nThreads));
			    //if (tempFuture.valid()) futures[i]=std::move(tempFuture);
				//futures.emplace_back(tempFuture);
				//futures.push_back(std::move(tempFuture));
						//task.wait();
			}
 */
			//std::cout<<threadPool.n_idle() <<" ";
			// for (int i=0; i<nThreads; i++) {
				// futures[i].wait();
			// }
			// threadPool.clear_queue();
			//threadPool.stop(true);
			//std::cout<<"after"<<B[!currentB][3*start]<<'\n';

			//for (int m=std::max(1,n+numCases-numSubjects); m<=std::min(n,numCases);m++) {


			// auto func = [&B,x,t,currentB,start,nThreads,nloop](const tbb::blocked_range<int>& range) {
			// auto func = [&B,x,t,currentB,start,nThreads,nloop](int i) {
			//
			//     int tStart = start + i * nloop / nThreads;
			//     int tEnd = start + ((i + 1) == nThreads ? nloop : (i + 1) * nloop / nThreads);
			//

			    for (int m = start; m <= end; ++m) {
			        T b = B[currentB][3*m-3];
			        T db = B[currentB][3*m-2];
			        T tb = t*b;
			        T xtb = x*tb;
			        T tdb = t*db;
			        B[!currentB][3*m] = B[currentB][3*m] + tb;
			        B[!currentB][3*m+1] = B[currentB][3*m+1] + tdb + xtb;
			        B[!currentB][3*m+2] = B[currentB][3*m+2] + t * B[currentB][3*m-1] + x*xtb + 2*x*tdb;
			    }
			// };


			// auto func = [&B,x,t,currentB](const tbb::blocked_range<int>& range) {
			//     for (int m = range.begin(); m < range.end(); ++m) {
			//         T b = B[currentB][3*m-3];
			//         T db = B[currentB][3*m-2];
			//         T tb = t*b;
			//         T xtb = x*tb;
			//         T tdb = t*db;
			//         B[!currentB][3*m] = B[currentB][3*m] + tb;
			//         B[!currentB][3*m+1] =  B[currentB][3*m+1] + tdb + xtb;
			//         B[!currentB][3*m+2] = B[currentB][3*m+2] + t * B[currentB][3*m-1] + x*xtb + 2*x*tdb;
			//     }
			// };
			//
			// tbb::parallel_for(
			//     tbb::blocked_range<int>(start, end+1), func
			//     );


			currentB = !currentB;
			++itExpXBeta;
			++itX;
			int m = std::min(n,numCases);
			for (int i=3*m+2; i >= 0; --i) {
				if (B[currentB][i] > 1e250) {
					//std::cout<<"resizing"<<'\n';
					for (int j=0; j<3*numCases+3; ++j) {
						B[currentB][j] /= 1e250;
					}
					break;
				}
			}
			/*
			if (B[currentB][3*m]>1e20 || B[currentB][3*m+1]>1e20 || B[currentB][3*m+2]>1e20) {
				for (int i=0; i<=numCases; i++) {
					B[currentB][3*i] /= 1e20;
					B[currentB][3*i+1] /= 1e20;
					B[currentB][3*i+2] /= 1e20;
				}
			}
			*/
		}

		result.push_back(B[currentB][3*numCases]);
		result.push_back(B[currentB][3*numCases+1]);
		result.push_back(B[currentB][3*numCases+2]);

	}

	//result.push_back(maxXi);
	//result.push_back(maxSorted);

	return result;
}

} // namespace

#endif /* RECURSIONS_HPP_ */
