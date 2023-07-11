#pragma once

#include "prox.hpp"

namespace rl::Proxs {

/*
 * Locally Low-Rank Regularizer
 *
 * The patch size is the volume over which the low-rank (SVD) calculation is performed.
 * Window size is the volume within this that is copied to the output. Set to 1 to get true sliding-window.
 */
struct LLR final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  float λ;
  Index patchSize, windowSize;
  Sz4   shape;
  bool  shift;
  LLR(float const, Index const, Index const, bool const, Sz4 const);

  void apply(float const α, CMap const &x, Map &z) const;
  void apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const;
};

} // namespace rl::Proxs
