#pragma once

#include "threads.hpp"

#include <ranges>
#include <tl/chunk.hpp>

namespace rl::Threads {
template <typename T, typename F> void ChunkFor(F f, std::vector<T> const &v)
{
  Index const    nt = GlobalThreadCount();
  Index const    cSz = std::ceil(v.size() / (float)nt); // Desired chunk size
  Index const    nC = std::ceil(v.size() / cSz);
  Eigen::Barrier barrier(static_cast<unsigned int>(nC));
  auto const     chunks = v | tl::views::chunk(cSz);
  for (auto const &chunk : chunks) {
    GlobalPool()->Schedule([&barrier, &f, chunk] {
      f(chunk);
      barrier.Notify();
    });
  }
  barrier.Wait();
}
} // namespace rl::Threads