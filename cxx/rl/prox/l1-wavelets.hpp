#pragma once

#include "../op/pad.hpp"
#include "../op/wavelets.hpp"
#include "norms.hpp"

namespace rl::Proxs {

struct L1Wavelets final : Prox
{
  PROX_INHERIT

  L1Wavelets(float const λ, Sz5 const shape, Index const width, std::vector<Index> const dims);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  Ops::Op::Ptr waves;
  L1           l1;
};

} // namespace rl::Proxs
