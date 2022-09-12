#pragma once

#include "types.h"
#include <functional>

// Forward declare
namespace Eigen {
class ThreadPoolDevice;
} // namespace Eigen

namespace rl {

namespace Threads {

Index GlobalThreadCount();
void SetGlobalThreadCount(Index n_threads);
Eigen::ThreadPoolDevice GlobalDevice();

using ForFunc = std::function<void(Index const index)>;
void For(ForFunc f, Index const n, std::string const &label);
void For(ForFunc f, Index const lo, Index const hi, std::string const &label);

} // namespace Threads
} // namespace rl
