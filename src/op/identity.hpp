#pragma once

#include "operator.hpp"

namespace rl {

template <typename Scalar_, size_t Rank>
struct IdentityOp : Operator<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)
  IdentityOp(Sz<Rank> dims)
    : Parent("IdentityOp", dims, dims)
  {
  }

  auto forward(InputMap x) const -> OutputMap
  {
    auto const time = Parent::startForward(x);
    Parent::finishAdjoint(x, time);
    return x;
  }

  auto adjoint(OutputMap x) const -> InputMap
  {
    auto const time = Parent::startAdjoint(x);
    Parent::finishAdjoint(x, time);
    return x;
  }
};

} // namespace rl
