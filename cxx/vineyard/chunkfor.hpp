#pragma once

#include "threads.hpp"

#include <ranges>
#include <tl/chunk.hpp>

namespace rl::Threads {
template <typename F, typename T, typename... Types> void ChunkFor(F f, std::vector<T> const &v, Types&... args)
{
  Index const    nt = GlobalThreadCount();
  Index const    cSz = std::ceil(v.size() / (float)nt); // Desired chunk size
  Index const    nC = std::ceil(v.size() / cSz);
  Eigen::Barrier barrier(static_cast<unsigned int>(nC));
  auto const     chunks = v | tl::views::chunk(cSz);
  for (auto const &chunk : chunks) {
    GlobalPool()->Schedule([&, chunk] {
      f(chunk, args...);
      barrier.Notify();
    });
  }
  barrier.Wait();
}
} // namespace rl::Threads