#pragma once

#include "../op/ops.hpp"
#include "common.hpp"
#include "givens.hpp"

namespace rl {

struct Bidiag
{
  using Op = Ops::Op;
  using Ptr = Op::Ptr;
  using Vector = Op::Vector;
  using Map = Op::Map;
  using CMap = typename Op::CMap;

  Bidiag(Ptr A, Ptr Minv, Ptr Ninv, Vector &x, CMap b, CMap x0);

  std::shared_ptr<Op> A;
  std::shared_ptr<Op> Minv, Ninv;
  Eigen::VectorXcf    u, Mu, v, Nv;
  float               α;
  float               β;

  void next();
};

} // namespace rl
