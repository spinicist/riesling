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
  void dimLoops(InMap &x, bool const rev) const;
  void wav1(Index const N, bool const rev, Cx1 &x) const;
  Index N_;
  Re1   Cc_, Cr_; // Coefficients
  Sz4   encodeDims_;
};
} // namespace rl
