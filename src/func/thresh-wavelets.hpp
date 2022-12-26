#pragma once

#include "thresh.hpp"
#include "op/wavelets.hpp"
#include "op/pad.hpp"

namespace rl {

struct ThresholdWavelets final : Prox<Cx4> {
    ThresholdWavelets(Sz4 const dims, float const λ, Index const width, Index const levels);
    auto operator()(float const α, Eigen::TensorMap<Cx4 const>) const -> Cx4;

private:
    Wavelets waves_;
    SoftThreshold<Cx4> thresh_;
};

} // namespace rl
