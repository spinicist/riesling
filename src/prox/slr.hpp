#pragma once

#include "fft/fft.hpp"
#include "prox.hpp"

namespace rl {

struct SLR final : Prox<Cx5>
{
  SLR(std::shared_ptr<FFT::FFT<5, 3>> const &fft, Index const kSz);
  std::shared_ptr<FFT::FFT<5, 3>> fft;
  Index kSz;

  void operator()(float const thresh, Vector const &x, Vector &z) const;
};
} // namespace rl
