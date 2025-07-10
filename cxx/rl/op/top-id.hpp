#pragma once

#include "top.hpp"

namespace rl::TOps {

template <int Rank> struct Identity : TOp<Rank, Rank>
{
  TOP_INHERIT(Rank, Rank)
  Identity(Sz<Rank> dims);

  void forward(InCMap x, OutMap y, float const s = 1.f) const;
  void adjoint(OutCMap y, InMap x, float const s = 1.f) const;

  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
};

} // namespace rl::TOps
