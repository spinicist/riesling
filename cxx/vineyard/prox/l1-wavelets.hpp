#pragma once

#include "op/pad.hpp"
#include "op/wavelets.hpp"
#include "prox/norms.hpp"

namespace rl::Proxs {

struct L1Wavelets final : Prox<Cx>
{
  PROX_INHERIT(Cx)

  L1Wavelets(float const λ, Sz4 const shape, Index const width, Sz4 const dims);
  void apply(float const α, CMap const &x, Map &z) const;

private:
  std::shared_ptr<Ops::Op<Cx>> waves_;
  L1                           thresh_;
};

} // namespace rl::Proxs
