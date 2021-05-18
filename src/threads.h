#pragma once

#include <functional>

// Forward declare
namespace Eigen {
class ThreadPoolDevice;
} // namespace Eigen

namespace Threads {
using ForFunc = std::function<void(long const index)>;
using RangeFunc = std::function<void(long const lo, long const hi)>;
using RangeThreadFunc = std::function<void(long const lo, long const hi, long const thread)>;

long GlobalThreadCount();
void SetGlobalThreadCount(long n_threads);
Eigen::ThreadPoolDevice GlobalDevice();

void For(ForFunc f, long const n);
void For(ForFunc f, long const lo, long const hi);
void RangeFor(RangeFunc f, long const n);
void RangeFor(RangeFunc f, long const lo, long const hi);
void RangeFor(RangeThreadFunc f, long const n);
void RangeFor(RangeThreadFunc f, long lo, long const hi);
} // namespace Threads
