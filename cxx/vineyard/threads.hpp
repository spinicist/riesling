#pragma once

#include "types.hpp"
#include <functional>

// Forward declare
namespace Eigen {
class ThreadPoolDevice;
} // namespace Eigen

namespace rl {

namespace Threads {

Eigen::ThreadPool       *GlobalPool();
Index                    GlobalThreadCount();
void                     SetGlobalThreadCount(Index n_threads);
Eigen::ThreadPoolDevice &GlobalDevice();

using ForFunc = std::function<void(Index const index)>;
void For(ForFunc f, Index const n, std::string const &label = "");
void For(ForFunc f, Index const lo, Index const hi, std::string const &label = "");

} // namespace Threads
} // namespace rl
