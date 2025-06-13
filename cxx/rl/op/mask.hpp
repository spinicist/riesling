#pragma once

#include "op.hpp"

namespace rl::Ops {

//! Mask (i.e. extract a set of elements from a vector)
template <typename Scalar = Cx> struct Mask final : Op<Scalar>
{
  OP_INHERIT
  using MaskVector = Eigen::Array<float, Eigen::Dynamic, 1>;
  Mask(MaskVector const &mask, Index const repeats);
  void  forward(CMap, Map) const;
  void  adjoint(CMap, Map) const;
  void  iforward(CMap x, Map y, float const s = 1.f) const;
  void  iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  MaskVector mask;
  Index repeats, isz, osz;
};
} // namespace rl::Ops
