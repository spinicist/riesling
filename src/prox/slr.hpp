#pragma once

#include "fft/fft.hpp"
#include "prox.hpp"

namespace rl::Prox {

struct SLR final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  SLR(float const, Index const, Sz5 const);
  void apply(float const α, CMap const &x, Map &z) const;

private:
  float λ;
  Index kSz;
  Sz5 shape;
  std::shared_ptr<FFT::FFT<5, 3>> fft;
};

} // namespace rl::Prox
