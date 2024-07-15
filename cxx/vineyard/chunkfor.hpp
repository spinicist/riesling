#pragma once

#include "threads.hpp"

#include <ranges>
#include <tl/chunk.hpp>

namespace rl::Threads {
template <typename T, typename F> void ChunkFor(F f, std::vector<T> const &v)
{
  Index const    nt = GlobalThreadCount();
  Eigen::Barrier barrier(static_cast<unsigned int>(nt));
  auto const     chunks = v | tl::views::chunk(nt);
  for (auto const &chunk : chunks) {
    fmt::print(stderr, "All: {} Chunk: {}\n", v.size(), chunk.size());
    GlobalPool()->Schedule([&barrier, &f, chunk] {
      f(chunk);
      fmt::print(stderr, "notifying\n");
      barrier.Notify();
    });
    fmt::print(stderr, "Scheduled a thread\n");
  }
  fmt::print(stderr, "WAITING\n");
  barrier.Wait();
  fmt::print(stderr, "Barrier said GO\n");
}
} // namespace rl::Threads