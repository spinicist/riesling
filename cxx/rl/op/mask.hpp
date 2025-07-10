#pragma once

#include "op.hpp"

namespace rl::Ops {

//! Mask (i.e. extract a set of elements from a vector)
struct Mask final : Op
{
  OP_INHERIT
  using MaskVector = Eigen::Array<float, Eigen::Dynamic, 1>;
  Mask(MaskVector const &mask, Index const repeats);
  static auto Make(MaskVector const &mask, Index const repeats) -> Ptr;
  void        forward(CMap, Map, float const) const;
  void        adjoint(CMap, Map, float const) const;
  void        iforward(CMap x, Map y, float const s = 1.f) const;
  void        iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  MaskVector mask;
  Index      repeats, isz, osz;
};
} // namespace rl::Ops
