/*
 * threads.cpp
 *
 * Copyright (c) 2019 Tobias Wood
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "threads.hpp"

#include "../log/log.hpp"

// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <Eigen/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace {
std::unique_ptr<Eigen::ThreadPool>           gp = nullptr;
std::unique_ptr<Eigen::CoreThreadPoolDevice> coreDev = nullptr;
std::unique_ptr<Eigen::ThreadPoolDevice>     tensorDev = nullptr;
} // namespace

namespace rl {
namespace Threads {

auto GlobalPool() -> Eigen::ThreadPool *
{
  if (gp == nullptr) {
    auto const nt = std::thread::hardware_concurrency();
    Log::Debug("Thread", "Creating default thread pool with {} threads", nt);
    gp = std::make_unique<Eigen::ThreadPool>(nt);
  }
  return gp.get();
}

auto GlobalThreadCount() -> Index { return GlobalPool()->NumThreads(); }

void SetGlobalThreadCount(Index nt)
{
  if (nt < 1) { nt = std::thread::hardware_concurrency(); }
  Log::Debug("Thread", "Creating thread pool with {} threads", nt);
  gp = std::make_unique<Eigen::ThreadPool>(nt);
  coreDev = std::make_unique<Eigen::CoreThreadPoolDevice>(*gp, nt);
  tensorDev = std::make_unique<Eigen::ThreadPoolDevice>(gp.get(), nt);
}

auto CoreDevice() -> Eigen::CoreThreadPoolDevice &
{
  if (coreDev == nullptr) {
    auto gp = GlobalPool();
    coreDev = std::make_unique<Eigen::CoreThreadPoolDevice>(*gp, gp->NumThreads());
  }
  return *coreDev;
}

auto TensorDevice() -> Eigen::ThreadPoolDevice &
{
  if (tensorDev == nullptr) {
    auto gp = GlobalPool();
    tensorDev = std::make_unique<Eigen::ThreadPoolDevice>(gp, gp->NumThreads());
  }
  return *tensorDev;
}

} // namespace Threads
} // namespace rl
