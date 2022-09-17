#pragma once

#include "functor.hpp"
#include "op/fft.hpp"

namespace rl {

struct SLR final : Functor<Cx5>
{
  SLR(FFTOp<5> const &fft, Index const kSz, float const thresh);
  FFTOp<5> const &fft;
  Index kSz;
  float thresh;

  auto operator()(Cx5 const &) const -> Cx5;
};
} // namespace rl
