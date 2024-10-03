#pragma once

#include "types.hpp"
#include <functional>
#include <span>

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

template <typename F, typename T, typename... Types> void ChunkFor(F const &f, std::vector<T> const &vv, Types &...args)
{
  std::span<T const> v{vv};
  Index const        nT = GlobalThreadCount();
  if (v.size() == 0) {
    return;
  } else {
    Index const    nC = std::min<Index>(v.size(), nT);
    Index const    den = v.size() / nC;
    Index const    rem = v.size() % nC;
    Eigen::Barrier barrier(nC);
    for (Index it = 0; it < nC; it++) {
      Index const        lo = it * den + std::min(it, rem);
      Index const        hi = (it + 1) * den + std::min(it + 1, rem);
      Index const        n = hi - lo;
      std::span<T const> sv = v.subspan(lo, n);
      GlobalPool()->Schedule([&, f, sv] {
        f(sv, args...);
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
}

template <typename F> void ChunkFor(F const &f, Index sz)
{
  Index const nT = GlobalThreadCount();
  if (sz == 0) {
    return;
  } else {
    Index const    den = sz / nT;
    Index const    rem = sz % nT;
    Index const    nC = std::min<Index>(sz, nT);
    Eigen::Barrier barrier(nC);
    for (Index it = 0; it < nC; it++) {
      Index const lo = it * den + std::min(it, rem);
      Index const hi = (it + 1) * den + std::min(it + 1, rem);
      GlobalPool()->Schedule([&barrier, f, lo, hi] {
        f(lo, hi);
        barrier.Notify();
      });
    }
    barrier.Wait();
  }
}

} // namespace Threads
} // namespace rl
