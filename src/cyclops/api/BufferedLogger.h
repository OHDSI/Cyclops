/*
 * BufferedLogger.h
 *
 * Host-agnostic implementations of the Cyclops logging/error abstractions.
 *
 * `bsccs::loggers::ProgressLogger` and `ErrorHandler` are the two hooks the core
 * uses to talk to its host language. The R package supplies Rcpp-backed versions
 * (`src/RcppProgressLogger.h`); this pair is the equivalent for any host that
 * speaks plain C++:
 *
 *   - progress lines are buffered and drained by the caller, so no assumption is
 *     made about where standard output goes;
 *   - errors become exceptions, matching `Rcpp::stop`'s non-returning contract.
 *
 * Both are safe to share across the worker threads Cyclops spawns for
 * cross-validation.
 */

#ifndef CYCLOPS_API_BUFFEREDLOGGER_H_
#define CYCLOPS_API_BUFFEREDLOGGER_H_

#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cyclops/api/CyclopsApi.h"
#include "cyclops/io/ProgressLogger.h"

namespace cyclops {
namespace api {

/// Collects progress output for the host to drain at its convenience.
class BufferedLogger : public bsccs::loggers::ProgressLogger {
public:
    /// Invoked from `yield()` on the thread that constructed the logger. Hosts
    /// use it to poll for cancellation (`PyErr_CheckSignals`, and so on) and may
    /// throw to abort the fit.
    using YieldHook = std::function<void()>;

    explicit BufferedLogger(bool silent = true)
        : silent_(silent), owner_(std::this_thread::get_id()) {}

    void writeLine(const std::ostringstream& stream) override {
        if (silent_) return;
        std::lock_guard<std::mutex> guard(mutex_);
        lines_.push_back(stream.str());
    }

    /// Called from the optimizer's inner loops; must be cheap.
    void yield() override {
        // Only poll on the thread that owns the host runtime: Cyclops calls
        // yield() from cross-validation workers too, and most host runtimes
        // (CPython included) only permit signal checks on the main thread.
        if (yield_hook_ && std::this_thread::get_id() == owner_) {
            yield_hook_();
        }
    }

    void setSilent(bool silent) override { silent_ = silent; }

    // Buffering is unconditional, so the core's concurrent/flush protocol needs
    // no special handling here.
    void setConcurrent(bool) override {}
    void flush() override {}

    void setYieldHook(YieldHook hook) { yield_hook_ = std::move(hook); }

    /// Removes and returns everything buffered so far.
    std::vector<std::string> drain() {
        std::lock_guard<std::mutex> guard(mutex_);
        std::vector<std::string> out;
        out.swap(lines_);
        return out;
    }

private:
    bool silent_;
    const std::thread::id owner_;
    std::mutex mutex_;
    std::vector<std::string> lines_;
    YieldHook yield_hook_;
};

/// Turns Cyclops' internal error reports into `CyclopsError` exceptions.
///
/// When the core is running concurrently it expects `throwError` to record the
/// message and return, with a later `flush()` raising; throwing out of a worker
/// thread would terminate the process instead.
class ThrowingErrorHandler : public bsccs::loggers::ErrorHandler {
public:
    void throwError(const std::ostringstream& stream) override {
        if (concurrent_) {
            std::lock_guard<std::mutex> guard(mutex_);
            deferred_.push_back(stream.str());
            return;
        }
        throw CyclopsError(stream.str());
    }

    void setConcurrent(bool concurrent) override { concurrent_ = concurrent; }

    void flush() override {
        std::string message;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (deferred_.empty()) return;
            for (const auto& line : deferred_) {
                message += line;
                message += '\n';
            }
            deferred_.clear();
        }
        throw CyclopsError(message);
    }

private:
    bool concurrent_ = false;
    std::mutex mutex_;
    std::vector<std::string> deferred_;
};

}  // namespace api
}  // namespace cyclops

#endif  // CYCLOPS_API_BUFFEREDLOGGER_H_
