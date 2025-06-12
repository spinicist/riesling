#pragma once

#include "prox.hpp"

#include "../op/fft.hpp"
#include "../op/hankel.hpp"

namespace rl::Proxs {

template<int NK>
struct SLR final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  SLR(float const, Sz5 const shape, Sz<NK> const dims, Sz<NK> const kW, bool const virt);
  void apply(float const α, CMap x, Map z) const;

private:
  float  λ;
  Sz5 shape;
  rl::TOps::FFT<5, 3> F;
  rl::TOps::Hankel<Cx, 5, NK> H;
};

} // namespace rl::Proxs
