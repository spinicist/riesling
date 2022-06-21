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
  Log::StartProgress((hi - lo)/GlobalPool()->NumThreads(), label);
  if (GlobalPool()->NumThreads() == 1) {
    for (Index i = lo; i < hi; i++) {
      f(i);
      Log::Tick();
    }
  } else {
    RangeThreadFunc rfunc = [&](Index const rlo, Index const rhi, Index const thread) {
      for (Index ii = rlo; ii < rhi; ii++) {
        f(ii);
        if (thread == 0) {
          Log::Tick();
        }
      }
    };
    RangeFor(rfunc, lo, hi);
  }
  Log::StopProgress();
}

void For(ForFunc f, Index const n, std::string const &label)
{
  For(f, 0, n, label);
}

void RangeFor(RangeFunc f, Index const n)
{
  RangeFor(f, 0, n);
}

void RangeFor(RangeFunc f, Index const lo, Index const hi)
{
  Index const nt = GlobalPool()->NumThreads();
  if (nt == 1) {
    f(lo, hi);
  } else {
    Index const ni = hi - lo;
    if (ni == 0) {
      return;
    }
    Index const num = std::min(nt, ni);
    Eigen::Barrier barrier(static_cast<unsigned int>(num));
    Index const range_sz = static_cast<Index>(std::ceil(static_cast<float>(ni) / num));
    Index range_lo = lo;
    Index range_hi = lo + range_sz;
    for (Index ti = 0; ti < num; ti++) {
      GlobalPool()->Schedule([&barrier, &f, range_lo, range_hi, hi] {
        f(range_lo, std::min(range_hi, hi));
        barrier.Notify();
      });
      range_lo += range_sz;
      range_hi += range_sz;
    }
    barrier.Wait();
  }
}

void RangeFor(RangeThreadFunc f, Index const n)
{
  RangeFor(f, 0, n);
}

void RangeFor(RangeThreadFunc f, Index const lo, Index const hi)
{
  Index const nt = GlobalPool()->NumThreads();
  Index const ni = hi - lo;
  if (ni == 0) {
    return;
  }
  Index const num = std::min(nt, ni);
  Eigen::Barrier barrier(static_cast<unsigned int>(num));
  Index const range_sz = static_cast<Index>(std::ceil(static_cast<float>(ni) / num));
  Index range_lo = lo;
  Index range_hi = lo + range_sz;
  for (Index ti = 0; ti < num; ti++) {
    GlobalPool()->Schedule([&barrier, &f, range_lo, range_hi, hi, ti] {
      f(range_lo, std::min(range_hi, hi), ti);
      barrier.Notify();
    });
    range_lo += range_sz;
    range_hi += range_sz;
  }
  barrier.Wait();
}

} // namespace Threads