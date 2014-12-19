#ifndef THREAD_TYPES_H_
#define THREAD_TYPES_H_

#include <thread>
#include <mutex>
#include "tinythread/tinythread.h" 

namespace bsccs {
#if defined(_WIN32) || defined(__WIN32__) || defined(__WINDOWS__) || defined(WIN_BUILD)
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

#if defined(_WIN32) || defined(__WIN32__) || defined(__WINDOWS__) || defined(WIN_BUILD)
    typedef threading::tthread_thread DefaultThreadType;
#else
    typedef threading::std_thread DefaultThreadType;
#endif

} // namespace bsccs

#endif // THREAD_TYPES_H_