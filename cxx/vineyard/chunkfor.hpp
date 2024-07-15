#pragma once

#include "threads.hpp"

#include <ranges>
#include <tl/chunk.hpp>

#include "log.hpp"

namespace rl::Threads {

template <typename F, typename T, typename... Types> void ChunkFor(F f, std::vector<T> const &v, Types&... args)
{
  if (v.size() == 0) {
    Log::Debug("No work to do");
    return;
  } else if (v.size() == 1) {
    f(v, args...);
  }
  Index const    nt = GlobalThreadCount();
  Index const    cSz = std::max(std::floor(v.size() / (float)nt), 1.f); // Desired chunk size
  Index const    nC = std::ceil(v.size() / cSz);
  Eigen::Barrier barrier(static_cast<unsigned int>(nC));
  auto const     chunks = v | tl::views::chunk(cSz);
  Log::Debug("Threads {} Chunks {} Size {}", nt, nC, cSz);
  for (auto const &chunk : chunks) {
    Log::Debug("Scheduling...");
    GlobalPool()->Schedule([&, chunk] {
      f(chunk, args...);
      Log::Debug("Notifying...");
      barrier.Notify();
    });
  }
  Log::Debug("Waiting...");
  barrier.Wait();
  Log::Debug("Finished");
}

} // namespace rl::Threads