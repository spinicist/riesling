#pragma once

#include "operator.hpp"

namespace rl {

struct Wavelets final : Operator<Cx, 4, 4>
{
  OP_INHERIT( Cx, 4, 4 )

  Wavelets(Sz4 const dims, Index const N, Index const levels);

  OP_DECLARE()

  static auto PaddedDimensions(Sz4 const dims) -> Sz4;
private:
  void encode_dim(InputMap image, Index const dim, Index const level) const;
  void decode_dim(OutputMap image, Index const dim, Index const level) const;
  Index N_;
  Re1 D_; // Coefficients
  Sz4 levels_;
};
} // namespace rl
