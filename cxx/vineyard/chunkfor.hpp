#pragma once

#include "threads.hpp"

#include <ranges>

namespace rl::Threads {
template <typename T, typename F> void ChunkFor(F f, std::vector<T> const &v)
{
  Index const    nt = GlobalThreadCount();
  Eigen::Barrier barrier(static_cast<unsigned int>(nt));
  auto const     chunks = v | std::views::chunk(nt);
  for (auto const &chunk : chunks) {
    GlobalPool()->Schedule([&barrier, &f, &chunk] {
      f(chunk);
      barrier.Notify();
    });
  }
  barrier.Wait();
}
} // namespace rl::Threads