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
#include "log.hpp"

// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace {
std::unique_ptr<Eigen::ThreadPool>       gp = nullptr;
std::unique_ptr<Eigen::ThreadPoolDevice> dev = nullptr;
} // namespace

namespace rl {
namespace Threads {

Eigen::ThreadPool *GlobalPool()
{
  if (gp == nullptr) {
    auto const nt = std::thread::hardware_concurrency();
    Log::Debug("Creating default thread pool with {} threads", nt);
    gp = std::make_unique<Eigen::ThreadPool>(nt);
  }
  return gp.get();
}

void SetGlobalThreadCount(Index nt)
{
  if (nt < 1) { nt = std::thread::hardware_concurrency(); }
  Log::Debug("Creating thread pool with {} threads", nt);
  gp = std::make_unique<Eigen::ThreadPool>(nt);
  dev = std::make_unique<Eigen::ThreadPoolDevice>(gp.get(), nt);
}

Index GlobalThreadCount() { return GlobalPool()->NumThreads(); }

Eigen::ThreadPoolDevice &GlobalDevice()
{
  if (dev == nullptr) {
    auto gp = GlobalPool();
    dev = std::make_unique<Eigen::ThreadPoolDevice>(gp, gp->NumThreads());
  }
  return *dev;
}

void For(ForFunc f, Index const lo, Index const hi, std::string const &label)
{
  Index const ni = hi - lo;
  Index const nt = GlobalPool()->NumThreads();
  if (ni == 0) { return; }

  bool const report = label.size();
  if (report) { Log::StartProgress(ni, label); }
  if (nt == 1) {
    for (Index ii = lo; ii < hi; ii++) {
      f(ii);
      if (report) { Log::Tick(); }
    }
  } else {
    Eigen::Barrier barrier(static_cast<unsigned int>(ni));
    for (Index ii = lo; ii < hi; ii++) {
      GlobalPool()->Schedule([&barrier, &f, ii, report] {
        f(ii);
        barrier.Notify();
        if (report) { Log::Tick(); }
      });
    }
    barrier.Wait();
  }
  if (report) { Log::StopProgress(); }
}

void For(ForFunc f, Index const n, std::string const &label) { For(f, 0, n, label); }

} // namespace Threads
} // namespace rl
