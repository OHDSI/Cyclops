#ifndef THREAD_TYPES_H_
#define THREAD_TYPES_H_

#include <algorithm>
#include <thread>
#include <mutex>
#include "tinythread/tinythread.h"

#if defined(_WIN32) || defined(__WIN32__) || defined(__WINDOWS__) || defined(WIN_BUILD)
    #define USE_TTHREAD
#else
    #undef USE_TTHREAD
#endif

namespace bsccs {
#ifdef USE_TTHREAD
    using tthread::mutex;
    using tthread::thread;
#else
    using std::mutex;
    using std::thread;
#endif

namespace threading {
    struct std_thread {};
    struct tthread_thread {};
} // threading

#ifdef USE_TTHREAD
    typedef threading::tthread_thread DefaultThreadType;
#else
    typedef threading::std_thread DefaultThreadType;
#endif

namespace threading {
namespace tthread {

    template <typename ItType, typename Function>
    struct for_each_arguments {
        ItType begin;
        ItType end;
        Function function;

        for_each_arguments(ItType begin, ItType end, Function function)
            : begin(begin), end(end), function(function) { }
    };

    template <typename ItType, typename Function>
    void for_each(void *a) {
       try {
           typedef for_each_arguments<ItType,Function> ArgType;
           auto args = bsccs::unique_ptr<ArgType>( // To avoid leaks
                static_cast<ArgType*>(a)
            );

            std::for_each(args->begin, args->end, args->function);
       } catch (...) {
       }
    }
}
}

template <typename InputIt>
struct TaskScheduler {

	TaskScheduler(InputIt begin, InputIt end, const size_t nThreads)
	   : begin(begin), end(end),
	     taskCount(std::distance(begin, end)),
	     nThreads(std::min(nThreads, taskCount)),
	     chunkSize(
	     	taskCount / nThreads + (taskCount % nThreads != 0)
	     ) { }

    template <typename UnaryFunction>
    UnaryFunction execute(UnaryFunction function) {
        return execute(function, DefaultThreadType());
    }

	size_t getThreadIndex(size_t i) {
		return nThreads == 1 ? 0 :
			i / chunkSize;
	}

	size_t getChunkSize() const { return chunkSize; }

	int getThreadCount() const { return nThreads; }

private:

#ifdef USE_TTHREAD
	template <typename UnaryFunction>
	UnaryFunction execute(UnaryFunction function, threading::tthread_thread) {

        std::vector<tthread::thread*> workers;
		size_t start = 0;
		for (size_t i = 0; i < nThreads - 1 && begin + start + chunkSize < end; ++i, start += chunkSize) {

            workers.emplace_back(new tthread::thread(
                threading::tthread::for_each<InputIt,UnaryFunction>,
                new threading::tthread::for_each_arguments<InputIt, UnaryFunction>(
                begin + start,
                begin + start + chunkSize,
                function)));
		}

		auto rtn = std::for_each(begin + start, end, function);
		for (size_t i = 0; i < workers.size(); ++i) {
			workers[i]->join();
			delete workers[i];
		}

		return rtn;
	}
#else
    template <typename UnaryFunction>
	UnaryFunction execute(UnaryFunction function, threading::std_thread) {

		std::vector<std::thread> workers;
		size_t start = 0;
		for (int i = 0; i < nThreads - 1 && begin + start + chunkSize < end; ++i, start += chunkSize) {

            workers.emplace_back(std::thread(
				std::for_each<InputIt, UnaryFunction>,
				begin + start,
				begin + start + chunkSize,
				function));
		}

		auto rtn = std::for_each(begin + start, end, function);
		for (int i = 0; i < workers.size(); ++i) {
			workers[i].join();
		}
		return rtn;
	}  // TODO Remove code-duplication between std::thread and tthread::thread versions
#endif

	const InputIt begin;
	const InputIt end;
	const size_t taskCount;
	const size_t nThreads;
	const size_t chunkSize;
};

} // namespace bsccs

#endif // THREAD_TYPES_H_
