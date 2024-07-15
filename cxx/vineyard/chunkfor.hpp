#pragma once

#include "threads.hpp"

#include <span>

#include "log.hpp"

namespace rl::Threads {

template <typename F, typename T, typename... Types> void ChunkFor(F f, std::vector<T> const &vv, Types &...args)
{
  std::span<T const> v{vv};
  Index const        nT = GlobalThreadCount();
  if (v.size() == 0) {
    Log::Debug("No work to do");
    return;
  } else if (nT == 1 || v.size() < nT) {
    f(v, args...);
  } else {
    Index const    den = v.size() / nT;
    Index const    rem = v.size() % nT;
    Eigen::Barrier barrier(static_cast<unsigned int>(nT));
    for (Index it = 0; it < nT; it++) {
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

} // namespace rl::Threads