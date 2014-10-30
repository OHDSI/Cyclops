
#ifndef PARALLELLOOPS_H_
#define PARALLELLOOPS_H_

#include <vector>
#include <thread>
#include <boost/iterator/counting_iterator.hpp>

namespace bsccs {

struct SerialOnly { };
struct OpenMP { };
struct Vanilla { };

struct C11Threads {
	
	C11Threads(int threads, size_t size = 100) : nThreads(threads), minSize(size) { }
	
	int nThreads;
	size_t minSize;
	
 };

namespace variants {

	namespace impl {

		template <typename InputIt, typename UnaryFunction, class Info>
		inline UnaryFunction for_each(InputIt begin, InputIt end, UnaryFunction function, 
				Info& info) {
			
			const int nThreads = info.nThreads;
			const size_t minSize = info.minSize;	
										
			if (nThreads > 1 && std::distance(begin, end) >= minSize) {				  
				std::vector<std::thread> workers(nThreads - 1);
				size_t chunkSize = std::distance(begin, end) / nThreads;
				size_t start = 0;
				for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
					workers[i] = std::thread(
						std::for_each<InputIt, UnaryFunction>,
						begin + start, 
						begin + start + chunkSize, 
						function);
				}
				auto rtn = std::for_each(begin + start, end, function);
				for (int i = 0; i < nThreads - 1; ++i) {
					workers[i].join();
				}
				return rtn;
			} else {				
				return std::for_each(begin, end, function);
			}
		}	
	
	} // namespace impl

    template <class InputIt, class UnaryFunction, class Specifics>
    inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, Specifics) {
        return std::for_each(first, last, f);    
    }
    
    template <class UnaryFunction, class Specifics>
    inline UnaryFunction for_each(int first, int last, UnaryFunction f, Specifics) {
        for (; first != last; ++first) {
            f(first);        
        }
        return f;
    }
    
    template <class UnaryFunction>
    inline UnaryFunction for_each(int first, int last, UnaryFunction f, C11Threads& x) {
    	return impl::for_each(boost::make_counting_iterator(first), boost::make_counting_iterator(last), f, x);
    }
    
#ifdef OPENMP    
    template <class UnaryFunction, class Specifics>
    inline UnaryFunction for_each(int first, int last, UnaryFunction f, OpenMP) {
        std::cout << "Parallel ";
        #pragma omp parallel for
        for (; first != last; ++first) {
            f(first);        
        }
        return f;
    }        
#endif     

} // namespace variants

} // namespace bsccs 

#endif // PARALLELLOOPS_H_