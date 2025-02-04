#pragma once

#include "top.hpp"

namespace rl::TOps {

template <typename Scalar_, int Rank> struct Identity : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  Identity(Sz<Rank> dims);

  void forward(InCMap const x, OutMap y) const;
  void adjoint(OutCMap const y, InMap x) const;

  void iforward(InCMap const x, OutMap y) const;
  void iadjoint(OutCMap const y, InMap x) const;
};

} // namespace rl::TOps
