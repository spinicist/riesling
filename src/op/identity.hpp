#pragma once

#include "operator.hpp"

namespace rl {

template <typename Scalar_, size_t Rank>
struct Identity : Operator<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)
  Identity(Sz<Rank> dims)
    : Parent("Identity", dims, dims)
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
