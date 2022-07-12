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

#include "threads.h"
#include "log.h"

// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace {
Eigen::ThreadPool *gp = nullptr;
}

namespace rl {
namespace Threads {

Eigen::ThreadPool *GlobalPool()
{
  if (gp == nullptr) {
    auto const nt = std::thread::hardware_concurrency();
    Log::Debug(FMT_STRING("Creating default thread pool with {} threads"), nt);
    gp = new Eigen::ThreadPool(nt);
  }
  return gp;
}

void SetGlobalThreadCount(Index nt)
{
  if (gp) {
    delete gp;
  }
  if (nt < 1) {
    nt = std::thread::hardware_concurrency();
  }
  Log::Debug(FMT_STRING("Creating new thread pool with {} threads"), nt);
  gp = new Eigen::ThreadPool(nt);
}

Index GlobalThreadCount()
{
  return GlobalPool()->NumThreads();
}

Eigen::ThreadPoolDevice GlobalDevice()
{
  return Eigen::ThreadPoolDevice(GlobalPool(), GlobalPool()->NumThreads());
}

void For(ForFunc f, Index const lo, Index const hi, std::string const &label)
{
  Index const ni = hi - lo;
  Index const nt = GlobalPool()->NumThreads();
  if (ni == 0) {
    return;
  }
  
  Log::StartProgress(ni, label);
  if (nt == 1) {
    for (Index ii = lo; ii < hi; ii++) {
      f(ii);
      Log::Tick();
    }
  } else {
    Eigen::Barrier barrier(static_cast<unsigned int>(ni));
    for (Index ii = lo; ii < hi; ii++) {
      GlobalPool()->Schedule([&barrier, &f, ii] {
        f(ii);
        barrier.Notify();
        Log::Tick();
      });
    }
    barrier.Wait();

  }
  Log::StopProgress();
}

void For(ForFunc f, Index const n, std::string const &label)
{
  For(f, 0, n, label);
}

} // namespace Threads
} // namespace rl
