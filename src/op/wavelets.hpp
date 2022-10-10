#pragma once

#include "operator.hpp"

namespace rl {

struct Wavelets final : Operator<4, 4>
{
  Wavelets(Sz4 const dims, Index const N, Index const levels);

  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output const &;
  auto adjoint(Output const &x) const -> Input const &;

  static auto PaddedDimensions(Sz4 const dims, Index const levels) -> Sz4;
private:
  void encode_dim(Input &image, Index const dim, Index const level) const;
  void decode_dim(Output &image, Index const dim, Index const level) const;
  Sz4 dims_;
  Index N_, L_;
  Re1 D_; // Coefficients
  mutable Input ws_;
};
} // namespace rl
