#pragma once

#include "types.hpp"

namespace rl {

template<int NDims>
struct Subgrid
{
  Sz<NDims>            gridSize, minCorner, maxCorner;
  std::vector<int32_t> indices;

  auto empty() const -> bool;
  auto count() const -> Index;
  auto size() const -> Sz<NDims>;
};

} // namespace rl
