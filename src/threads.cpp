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

// Need to define EIGEN_USE_THREADS before including these. This is done in CMakeLists.txt
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace {
Eigen::ThreadPool *gp = nullptr;
}

namespace Threads {

Eigen::ThreadPool *GlobalPool()
{
  if (gp == nullptr) {
    gp = new Eigen::ThreadPool(std::thread::hardware_concurrency());
  }
  return gp;
}

void SetGlobalThreadCount(long nT)
{
  if (gp) {
    delete gp;
  }
  gp = new Eigen::ThreadPool(nT);
}

long GlobalThreadCount()
{
  return GlobalPool()->NumThreads();
}

Eigen::ThreadPoolDevice GlobalDevice()
{
  return Eigen::ThreadPoolDevice(GlobalPool(), GlobalPool()->NumThreads());
}

void For(TFunc f, long const lo, long const hi)
{
  long const barrier_size = hi - lo;
  Eigen::Barrier barrier(static_cast<unsigned int>(barrier_size));
  for (long i = lo; i < hi; i++) {
    GlobalPool()->Schedule([&barrier, &f, i] {
      f(i);
      barrier.Notify();
    });
  }
  barrier.Wait();
}

void For(TFunc f, long const n)
{
  For(f, 0, n);
}

void RangeFor(RangeFunc f, long const n)
{
  RangeFor(f, 0, n);
}

void RangeFor(RangeFunc f, long const lo, long const hi)
{
  long const nt = GlobalPool()->NumThreads();
  long const ni = hi - lo;
  long const num = std::min(nt, ni);
  Eigen::Barrier barrier(static_cast<unsigned int>(num));
  long const range_sz = static_cast<long>(std::ceil(static_cast<float>(ni) / num));
  long range_lo = lo;
  long range_hi = lo + range_sz;
  for (long ti = 0; ti < num; ti++) {
    GlobalPool()->Schedule([&barrier, &f, range_lo, range_hi, hi] {
      f(range_lo, std::min(range_hi, hi));
      barrier.Notify();
    });
    range_lo += range_sz;
    range_hi += range_sz;
  }
  barrier.Wait();
}

} // namespace Threads