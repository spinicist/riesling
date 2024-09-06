#pragma once

#include "types.hpp"
#include <functional>

// Forward declare
namespace Eigen {
class CoreThreadPoolDevice;
} // namespace Eigen

namespace rl {

namespace Threads {

auto GlobalPool() -> Eigen::ThreadPool *;
auto CoreDevice() -> Eigen::CoreThreadPoolDevice &;
auto TensorDevice() -> Eigen::ThreadPoolDevice &;

auto GlobalThreadCount() -> Index;
void SetGlobalThreadCount(Index n_threads);

using ForFunc = std::function<void(Index const index)>;
void For(ForFunc f, Index const n, std::string const &label = "");
void For(ForFunc f, Index const lo, Index const hi, std::string const &label = "");

} // namespace Threads
} // namespace rl
