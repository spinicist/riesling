#pragma once

#include "op/pad.hpp"
#include "op/wavelets.hpp"
#include "thresh.hpp"

namespace rl::Proxs {

struct ThresholdWavelets final : Prox<Cx>
{
  PROX_INHERIT(Cx)

  ThresholdWavelets(float const λ, Sz4 const shape, Index const width, Index const levels);
  void apply(float const α, CMap const &x, Map &z) const;

private:
  std::shared_ptr<Ops::Op<Cx>> waves_;
  SoftThreshold                thresh_;
};

} // namespace rl::Proxs
