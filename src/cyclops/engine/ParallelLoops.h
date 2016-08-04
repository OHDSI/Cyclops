
#ifndef PARALLELLOOPS_H_
#define PARALLELLOOPS_H_

#include <vector>
#include <numeric>
#include <thread>
#include <boost/iterator/counting_iterator.hpp>

#pragma GCC diagnostic push
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic ignored "-Wpragmas"
#endif
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wignored-attributes" // To keep C++14 quiet
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "RcppParallel.h"
#pragma GCC diagnostic pop

//#include "engine/ThreadPool.h"

namespace bsccs {

struct SerialOnly { };
struct ParallelInfo { };
struct OpenMP { };
struct Vanilla { };
struct RcppParallel { };

struct C11Threads {

	C11Threads(int threads, size_t size = 100) : nThreads(threads), minSize(size) { }

	int nThreads;
	size_t minSize;
};

// struct C11ThreadPool {
//
// 	C11ThreadPool(int poolSize, int threads, size_t size = 100) : pool(poolSize), nThreads(threads), minSize(size) { }
// 	virtual ~C11ThreadPool() { };
//
// 	ThreadPool pool;
//
// 	int nThreads;
// 	size_t minSize;
// };


namespace variants {


    namespace trial {

        template <typename OuterResultType, typename InnerResultType,
                  typename OuterFunction, typename InnerFunction,
                  typename KeyIterator, typename InnerIterator, typename OuterIterator>
        inline OuterResultType nested_reduce(KeyIterator key, KeyIterator end,
                    InnerIterator inner, OuterIterator outer,
                    InnerResultType reset_in, OuterResultType result_out,
                    InnerFunction f_in, OuterFunction f_out) {

            const auto stop = end - 1;

            InnerResultType result_in = reset_in;

            for (; key != stop; ++key, ++inner) {

                result_in = f_in(result_in, *inner);

                if (*key != *(key + 1)) {

                    result_out = f_out(result_out, result_in, *outer);

                    result_in = reset_in;
                    ++outer;
                }
            }

            result_in = f_in(result_in, *inner);

            return f_out(result_out, result_in, *outer);
        }



    } // namespace trial

    const int nThreads = 4;
	const int minSize = 100000;

	namespace impl {


	    template <typename InputIt, typename ResultType, typename BinaryFunction>
	    struct Reducer : public ::RcppParallel::Worker {

	        Reducer(InputIt begin, InputIt end, ResultType result, BinaryFunction function) : begin(begin), end(end), result(result), function(function) {}
	        Reducer(const Reducer& rhs, ::RcppParallel::Split) : begin(rhs.begin), end(rhs.end), result(0), function(rhs.function) { }

	        void operator()(std::size_t i, std::size_t j) {
	            result = std::accumulate(begin + i, begin + j, result, function);
	        }

	        void join(const Reducer& rhs) {
	            result += rhs.result;
	        }

	        InputIt begin;
	        InputIt end;
	        ResultType result;
	        BinaryFunction function;

	    };

		template <typename InputIt, typename UnaryFunction>
		struct WrapWorker : public ::RcppParallel::Worker {

			WrapWorker(InputIt begin, InputIt end, UnaryFunction function) :
					begin(begin), end(end), function(function) { }

			void operator()(std::size_t i, std::size_t j) {
				std::for_each(begin + i, begin + j, function);
			}

			InputIt begin;
			InputIt end;
			UnaryFunction function;
		};

		template <typename InputIt, typename ResultType, typename BinaryFunction>
		inline ResultType reduce(InputIt begin, InputIt end, ResultType result, BinaryFunction function, RcppParallel& tbb) {

		Reducer<InputIt,ResultType,BinaryFunction> reducer(begin, end, result, function);

		::RcppParallel::parallelReduce(0, std::distance(begin,end), reducer);

		return reducer.result;
		}

		template <typename InputIt, typename UnaryFunction>
		inline UnaryFunction for_each(InputIt begin, InputIt end, UnaryFunction function,
				RcppParallel& tbb) {

			auto worker = WrapWorker<InputIt, UnaryFunction>(begin, end, function);

			::RcppParallel::parallelFor(0, std::distance(begin, end), worker);

			return function;
		}

// 		template <typename InputIt, typename UnaryFunction>
// 		inline UnaryFunction for_each(InputIt begin, InputIt end, UnaryFunction function,
// 				C11Threads& info) {
//
// // 			const int nThreads = info.nThreads;
// // 			const size_t minSize = info.minSize;
//
// // 			std::cout << "I";
//
// 			if (nThreads > 1 && std::distance(begin, end) >= minSize) {
// 				std::vector<std::thread> workers(nThreads - 1);
// 				size_t chunkSize = std::distance(begin, end) / nThreads;
// 				size_t start = 0;
// 				for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
// 					workers[i] = std::thread(
// 						std::for_each<InputIt, UnaryFunction>,
// 						begin + start,
// 						begin + start + chunkSize,
// 						function);
// 				}
// 				auto rtn = std::for_each(begin + start, end, function);
// 				for (int i = 0; i < nThreads - 1; ++i) {
// 					workers[i].join();
// 				}
// // 				std::cout << "P";
// 				return rtn;
// 			} else {
// // 				std::cout << "N";
// 				return std::for_each(begin, end, function);
// 			}
// 		}

#if 0
		template <typename InputIt, typename UnaryFunction>
		inline UnaryFunction for_each(InputIt begin, InputIt end, UnaryFunction function,
				C11ThreadPool& tpool) {

			const int nThreads = tpool.nThreads;
 			const size_t minSize = tpool.minSize;

 			if (nThreads > 1 && std::distance(begin, end) >= minSize) {

				std::vector< std::future<void> > results;

				size_t chunkSize = std::distance(begin, end) / nThreads;
				size_t start = 0;

//   				std::cout << "Start!" << std::endl;



				for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
					results.emplace_back(
						tpool.pool.enqueue([=] {
							std::for_each(
								begin + start,
								begin + start + chunkSize,
								function);
						})
					);
				}
				results.emplace_back(
					tpool.pool.enqueue([=] {
						std::for_each(begin + start, end, function);
					})
				);
// 				auto rtn = std::for_each(begin + start, end, function);

// 				for (int i = 0; i < nThreads - 1; ++i) {
// 					workers[i].join();
// 				}
				for (auto&& result: results) result.get();

//  				std::cout << "Done!" << std::endl;

				return function;
			} else {
				return std::for_each(begin, end, function);
			}
		}
#endif


	} // namespace impl


//     template <class InputIt, class UnaryFunction, class Specifics>
//     inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, Specifics) {
//         return std::for_each(first, last, f);
//     }

    template <class InputIt, class UnaryFunction>
    inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, SerialOnly x) {
        return std::for_each(first, last, f);
    }

//     template <class UnaryFunction, class Specifics>
//     inline UnaryFunction for_each(int first, int last, UnaryFunction f, Specifics) {
//         for (; first != last; ++first) {
//             f(first);
//         }
//         return f;
//     }

    template <class InputIt, class UnaryFunction>
    inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, C11Threads& x) {
        return impl::for_each(first, last, f, x);
    }

//    template <class InputIt, class UnaryFunction>
//    inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, C11ThreadPool& x) {
//        return impl::for_each(first, last, f, x);
//    }

    template <class InputIt, class UnaryFunction>
    inline UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f, RcppParallel x) {
        return impl::for_each(first, last, f, x);
    }
//
//     template <class UnaryFunction>
//     inline UnaryFunction for_each(int first, int last, UnaryFunction f, C11Threads& x) {
//     	return impl::for_each(boost::make_counting_iterator(first), boost::make_counting_iterator(last), f, x);
//     }
//
//     template <class UnaryFunction>
//     inline UnaryFunction for_each(int first, int last, UnaryFunction f, C11ThreadPool& x) {
//     	return impl::for_each(boost::make_counting_iterator(first), boost::make_counting_iterator(last), f, x);
//     }

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

		template <class InputIt, class ResultType, class BinaryFunction>
		struct Reducer {
		    void operator()(InputIt begin, InputIt end, ResultType x0, BinaryFunction function,
		            ResultType& result) {
		        result = std::accumulate(begin, end, x0, function);
		    }
		};


		template <class InputIt, class ResultType, class BinaryFunction>
		inline ResultType reduce(InputIt begin, InputIt end,
		    ResultType result, BinaryFunction function,
		    RcppParallel tbb) {


				return impl::reduce(begin, end, result, function, tbb);


		}

    	template <class InputIt, class ResultType, class BinaryFunction>
	    inline ResultType reduce(InputIt begin, InputIt end,
	            ResultType result, BinaryFunction function, SerialOnly) {
	        return std::accumulate(begin, end, result, function);
	    }

//     	template <class InputIt, class ResultType, class BinaryFunction, class Info>
// 	    inline ResultType reduce(InputIt begin, InputIt end,
// 	            ResultType result, BinaryFunction function, Info& info) {
//
// // 	        const int nThreads = info.nThreads;
// // 	        const size_t minSize = info.minSize;
//
// 	        if (nThreads > 1 && std::distance(begin, end) >= minSize) {
// // 	            std::cout << "PR" << std::endl;
//
// 	            std::vector<std::thread> workers(nThreads - 1);
// 	            std::vector<ResultType> fractions(nThreads - 1);
//
// 	            size_t chunkSize = std::distance(begin, end) / nThreads;
// 	            size_t start = 0;
// 	            for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
// 	                workers[i] = std::thread(
// 	                    Reducer<InputIt, ResultType, BinaryFunction>(),
// 	                    begin + start,
// 	                    begin + start + chunkSize,
// 	                    ResultType(), function,
// 	                    std::ref(fractions[i])
// 	                    );
// // 	                std::cout << std::distance(begin + start, begin + start + chunkSize) << " ";
// 	            }
//
// 	            result = std::accumulate(begin + start, end, result, function);
// // 	            std::cout << std::distance(begin + start, end) << std::endl;
// 	            for (int i = 0; i < nThreads - 1; ++i) {
// 	                workers[i].join();
// 	                result += fractions[i];
// 	            }
//
// 	            return result;
//
// 	        } else {
// 	            return std::accumulate(begin, end, result, function);
// 	        }
// 	    }

	    template <class IndexIt, class OutputIt, class Transform>
	    inline void transform_segmented_reduce(IndexIt i, IndexIt end,
	            IndexIt j,
	            Transform transform,
	            const OutputIt x,
	            OutputIt y, SerialOnly info) {
	    	for (; i != end; ++i, ++j) {
	    		y[i] += transform(x[j], j);
	    	}
	    }


// spmv_coo_serial_kernel(const unsigned int * I,
//                        const unsigned int * J,
// #ifndef IS_INDICATOR_MATRIX
//                        const REAL * V,
// #endif
//                        const REAL * x,
//                              REAL * y,
//                        const unsigned int num_nonzeros)
// {
//     for(unsigned int n = 0; n < num_nonzeros; n++){
// #ifdef IS_INDICATOR_MATRIX
//     	y[I[n]] += x[J[n]];
// #else
//     	y[I[n]] += V[n] * x[J[n]];
// #endif
//     }
// }


} // namespace variants

} // namespace bsccs

#endif // PARALLELLOOPS_H_
