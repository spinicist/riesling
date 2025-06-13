#pragma once

#include "top.hpp"

namespace rl::TOps {

template <typename Scalar_, int Rank> struct Identity : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  Identity(Sz<Rank> dims);

  void forward(InCMap x, OutMap y) const;
  void adjoint(OutCMap y, InMap x) const;

  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
};

} // namespace rl::TOps
