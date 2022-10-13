#pragma once

#include "thresh.hpp"
#include "op/wavelets.hpp"
#include "op/pad.hpp"

namespace rl {

struct ThresholdWavelets final : Prox<Cx4> {
    ThresholdWavelets(Sz4 const dims, Index const width, Index const levels);
    auto operator()(float const λ, Cx4 const &) const -> Cx4 const &;

private:
    PadOp<4> pad_;
    Wavelets waves_;
    SoftThreshold thresh_;
};

} // namespace rl
