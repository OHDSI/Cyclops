/*
 * WorkerPool.h
 *
 * Persistent threads for short parallel loops. run(count, task) calls task(i, thread) for every
 * i in [0, count), on the calling thread (thread 0) and on the workers (1 .. size() - 1), and
 * returns when all calls are done. Tasks are handed out one at a time, so uneven tasks balance.
 * Runs come in quick succession, so workers and the caller spin briefly before sleeping.
 */

#ifndef WORKERPOOL_H_
#define WORKERPOOL_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace bsccs {

class WorkerPool {
public:
    typedef std::function<void(size_t, int)> Task;

    explicit WorkerPool(int threads) {
        for (int thread = 1; thread < threads; ++thread) {
            workers.emplace_back([this, thread] { work(thread); });
        }
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping.store(true);
        }
        start.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    int size() const { return static_cast<int>(workers.size()) + 1; }

    void run(size_t count, const Task& task) {
        if (workers.empty() || count < 2) {
            for (size_t i = 0; i < count; ++i) {
                task(i, 0);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            job = &task;
            jobCount = count;
            next.store(0);
            pending.store(static_cast<int>(workers.size()));
            error = nullptr;
            generation.fetch_add(1);
        }
        start.notify_all();
        execute(0);

        spin([this] { return pending.load() == 0; });
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [this] { return pending.load() == 0; });
        job = nullptr;
        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    void execute(int thread) {
        try {
            for (size_t i = next.fetch_add(1); i < jobCount; i = next.fetch_add(1)) {
                (*job)(i, thread);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!error) {
                error = std::current_exception();
            }
            next.store(jobCount); // skip the remaining tasks
        }
    }

    template <class Condition>
    static void spin(Condition condition) { // for up to 50 microseconds
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
        while (!condition() && std::chrono::steady_clock::now() < until) { }
    }

    void work(int thread) {
        size_t seen = 0;
        while (true) {
            spin([&] { return stopping.load() || generation.load() != seen; });
            {
                std::unique_lock<std::mutex> lock(mutex);
                start.wait(lock, [&] { return stopping.load() || generation.load() != seen; });
                if (stopping.load()) {
                    return;
                }
                seen = generation.load();
            }
            execute(thread);
            if (pending.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lock(mutex);
                done.notify_one();
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable start;
    std::condition_variable done;
    const Task* job = nullptr;
    size_t jobCount = 0;
    std::atomic<size_t> next{0};
    std::atomic<size_t> generation{0};
    std::atomic<int> pending{0};
    std::atomic<bool> stopping{false};
    std::exception_ptr error;
};

} // namespace bsccs

#endif /* WORKERPOOL_H_ */
