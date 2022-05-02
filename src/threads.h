#pragma once

#include "types.h"
#include <functional>

// Forward declare
namespace Eigen {
class ThreadPoolDevice;
} // namespace Eigen

namespace Threads {
using ForFunc = std::function<void(Index const index)>;
using RangeFunc = std::function<void(Index const lo, Index const hi)>;
using RangeThreadFunc = std::function<void(Index const lo, Index const hi, Index const thread)>;

Index GlobalThreadCount();
void SetGlobalThreadCount(Index n_threads);
Eigen::ThreadPoolDevice GlobalDevice();

void For(ForFunc f, Index const n, std::string const &label);
void For(ForFunc f, Index const lo, Index const hi, std::string const &label);
void RangeFor(RangeFunc f, Index const n);
void RangeFor(RangeFunc f, Index const lo, Index const hi);
void RangeFor(RangeThreadFunc f, Index const n);
void RangeFor(RangeThreadFunc f, Index lo, Index const hi);
} // namespace Threads
