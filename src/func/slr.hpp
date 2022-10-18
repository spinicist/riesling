#pragma once

#include "functor.hpp"
#include "op/fft.hpp"

namespace rl {

struct SLR final : Prox<Cx5>
{
  SLR(FFTOp< 5, 3> const &fft, Index const kSz);
  FFTOp< 5, 3> const &fft;
  Index kSz;

  auto operator()(float const thresh, Eigen::TensorMap<Cx5 const> x) const -> Eigen::TensorMap<Cx5>;
};
} // namespace rl
