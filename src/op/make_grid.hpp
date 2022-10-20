#pragma once

#include "trajectory.hpp"
#include "operator.hpp"
#include "gridBase.hpp"

#include <memory>
#include <optional>

namespace rl {

template <typename Scalar, size_t ND>
std::shared_ptr<GridBase<Scalar, ND>> make_grid(
  Trajectory const &trajectory,
  std::string const kType,
  float const os,
  Index const nC,
  std::optional<Re2> const &basis = std::nullopt);

} // namespace rl
