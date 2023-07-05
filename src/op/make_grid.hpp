#pragma once

#include "trajectory.hpp"
#include "ops.hpp"
#include "gridBase.hpp"

#include <memory>
#include <optional>

namespace rl {

auto IdBasis() -> Re2;

template <typename Scalar, size_t ND>
std::shared_ptr<GridBase<Scalar, ND>> make_grid(
  Trajectory const &trajectory,
  std::string const kType,
  float const os,
  Index const nC,
  Re2 const &basis = IdBasis());

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_3d_grid(
  Trajectory const &trajectory,
  std::string const kType,
  float const os,
  Index const nC,
  Re2 const &basis = IdBasis());

} // namespace rl
