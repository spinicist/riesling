#pragma once

#include "prox.hpp"

#include "../op/fft.hpp"
#include "../op/hankel.hpp"

namespace rl::Proxs {

/*
 * An attemp to guess the Proximal operator for enforcing Hermitian symmetry in k-space, i.e.
 * the prox to minimise λ|Fx - (Fx)†|
 */
struct Hermitian final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  Hermitian(float const, Sz5 const shape);
  void apply(float const α, CMap x, Map z) const;

private:
  float  λ;
  Sz5 shape;
  rl::TOps::FFT<5, 3> F;
};

} // namespace rl::Proxs
