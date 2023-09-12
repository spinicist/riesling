#pragma once

#include "tensorop.hpp"

namespace rl {

struct Wavelets final : TensorOperator<Cx, 4, 4>
{
  OP_INHERIT(Cx, 4, 4)

  Wavelets(Sz4 const shape, Index const N, Sz4 const dims);

  OP_DECLARE()

  static auto PaddedShape(Sz4 const shape, Sz4 const dims) -> Sz4;

private:
  void  encode_dim(OutMap &y, Index const dim, Index const level) const;
  void  decode_dim(InMap &x, Index const dim, Index const level) const;
  Index N_;
  Re1   D_; // Coefficients
  Sz4   levels_;
};
} // namespace rl
