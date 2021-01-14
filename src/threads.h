#pragma once

#include <functional>

// Forward declare
namespace Eigen {
class ThreadPoolDevice;
} // namespace Eigen

namespace Threads {
typedef std::function<void(long const index)> TFunc;
typedef std::function<void(long const lo, long const hi)> RangeFunc;

long GlobalThreadCount();
void SetGlobalThreadCount(long n_threads);
Eigen::ThreadPoolDevice GlobalDevice();

void For(TFunc f, long const n);
void For(TFunc f, long const lo, long const hi);
void RangeFor(RangeFunc f, long const n);
void RangeFor(RangeFunc f, long const lo, long const hi);

} // namespace Threads
