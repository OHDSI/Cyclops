#ifndef THREAD_TYPES_H_
#define THREAD_TYPES_H_

#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <exception>
#include <vector>
#include <cfenv>
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

    // Windows threads start with the system's floating-point control word, not the caller's (R sets
    // 64-bit x87 precision), so the arguments, built in the calling thread, carry its floating-point
    // environment for the worker to install before it computes anything.
    template <typename ItType, typename Function>
    struct for_each_arguments {
        ItType begin;
        ItType end;
        Function function;
        std::fenv_t environment;

        for_each_arguments(ItType begin, ItType end, Function function)
            : begin(begin), end(end), function(function) {
            std::fegetenv(&environment);
        }
    };

    template <typename ItType, typename Function>
    void for_each(void *a) {
       try {
           typedef for_each_arguments<ItType,Function> ArgType;
           auto args = bsccs::unique_ptr<ArgType>( // To avoid leaks
                static_cast<ArgType*>(a)
            );

            std::fesetenv(&args->environment);
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

#ifdef USE_TTHREAD
namespace threading {
namespace tthread {
    // Carries the calling thread's floating-point environment, as for_each_arguments does
    template <typename Worker>
    struct worker_arguments {
        Worker* worker;
        size_t index;
        std::fenv_t environment;

        worker_arguments(Worker* worker, size_t index) : worker(worker), index(index) {
            std::fegetenv(&environment);
        }
    };

    template <typename Worker>
    void run_worker(void *a) {
        try {
            auto args = bsccs::unique_ptr<worker_arguments<Worker>>(
                static_cast<worker_arguments<Worker>*>(a));
            std::fesetenv(&args->environment);
            (*args->worker)(args->index);
        } catch (...) {
        }
    }
}
}
#endif

// Runs function(task, worker) for task = 0, ..., taskCount - 1 on min(nThreads, taskCount) workers;
// each worker takes the next unstarted task when it finishes one, and worker 0 is the calling thread.
// The first exception thrown by any task stops the others from starting new tasks and is rethrown in
// the calling thread once all workers have finished.
template <typename BinaryFunction>
void executeDynamic(size_t taskCount, size_t nThreads, BinaryFunction function) {

    nThreads = std::max(static_cast<size_t>(1), std::min(nThreads, taskCount));
    std::atomic<size_t> next(0);
    std::atomic<bool> failed(false);
    std::exception_ptr failure; // written only by the worker that sets failed, read after the joins
    auto worker = [&next, &failed, &failure, taskCount, &function](size_t index) {
        try {
            for (size_t task = next++; task < taskCount; task = next++) {
                function(task, index);
            }
        } catch (...) {
            bool expected = false;
            if (failed.compare_exchange_strong(expected, true)) {
                failure = std::current_exception();
            }
            next = taskCount;
        }
    };

#ifdef USE_TTHREAD
    std::vector<tthread::thread*> workers;
    for (size_t i = 1; i < nThreads; ++i) {
        workers.emplace_back(new tthread::thread(
            threading::tthread::run_worker<decltype(worker)>,
            new threading::tthread::worker_arguments<decltype(worker)>{&worker, i}));
    }
    worker(0);
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i]->join();
        delete workers[i];
    }
#else
    std::vector<std::thread> workers;
    for (size_t i = 1; i < nThreads; ++i) {
        workers.emplace_back(worker, i);
    }
    worker(0);
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i].join();
    }
#endif
    if (failure) {
        std::rethrow_exception(failure);
    }
}

} // namespace bsccs

#endif // THREAD_TYPES_H_
