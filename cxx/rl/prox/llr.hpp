#pragma once

#include "prox.hpp"

namespace rl::Proxs {

/*
 * Locally Low-Rank Regularizer
 *
 * The patch size is the volume over which the low-rank (SVD) calculation is performed.
 * Window size is the volume within this that is copied to the output. Set to 1 to get true sliding-window.
 */
struct LLR final : Prox
{
  PROX_INHERIT
  float λ;
  Index patchSize, windowSize;
  Sz5   shape;
  bool  shift;
  LLR(float const, Index const, Index const, bool const, Sz5 const);

  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;
};

} // namespace rl::Proxs
