#pragma once

#include "op.hpp"

namespace rl::Ops {

//! Mask (i.e. extract a set of elements from a vector)
template <typename Scalar = Cx> struct Mask final : Op<Scalar>
{
  OP_INHERIT
  using MaskVector = Eigen::Array<float, Eigen::Dynamic, 1>;
  Mask(MaskVector const &mask, Index const repeats);
  void  forward(CMap const, Map) const;
  void  adjoint(CMap const, Map) const;
  void  iforward(CMap const x, Map y) const;
  void  iadjoint(CMap const y, Map x) const;

private:
  MaskVector mask;
  Index repeats, isz, osz;
};
} // namespace rl::Ops
