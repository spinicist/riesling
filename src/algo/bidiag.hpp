#pragma once

#include "common.hpp"
#include "op/operator.hpp"

namespace rl {

auto StableGivens(float const a, float const b) -> std::tuple<float, float, float>;
auto Rotation(float const a, float const b) -> std::tuple<float, float, float>;

void BidiagInit(
  std::shared_ptr<LinOps::Op<Cx>> op,
  std::shared_ptr<LinOps::Op<Cx>> M,
  Eigen::VectorXcf &Mu,
  Eigen::VectorXcf &u,
  Eigen::VectorXcf &v,
  float &α,
  float &β,
  Eigen::VectorXcf &x,
  Eigen::Map<Eigen::VectorXcf> const &b,
  Cx *x0);

void Bidiag(
  std::shared_ptr<LinOps::Op<Cx>> const op,
  std::shared_ptr<LinOps::Op<Cx>> const M,
  Eigen::VectorXcf &Mu,
  Eigen::VectorXcf &u,
  Eigen::VectorXcf &v,
  float &α,
  float &β);

} // namespace rl
