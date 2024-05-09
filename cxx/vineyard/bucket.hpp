#pragma once

#include "types.hpp"

namespace rl {

template<int NDims>
struct Bucket
{
  Sz<NDims>            gridSize, minCorner, maxCorner;
  std::vector<int32_t> indices;

  auto empty() const -> bool;
  auto size() const -> Index;
  auto bucketSize() const -> Sz<NDims>;
  auto bucketStart() const -> Sz<NDims>;
  auto gridStart() const -> Sz<NDims>;
  auto sliceSize() const -> Sz<NDims>;
};

} // namespace rl
