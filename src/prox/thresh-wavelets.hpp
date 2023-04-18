#pragma once

#include "thresh.hpp"
#include "op/wavelets.hpp"
#include "op/pad.hpp"

namespace rl {

struct ThresholdWavelets final : Prox<Cx> {
    PROX_INHERIT(Cx)

    ThresholdWavelets(float const λ, Sz4 const shape, Index const width, Index const levels);
    void apply(float const α, CMap const &x, Map &z) const;

private:
    std::shared_ptr<LinOps::Op<Cx>> waves_;
    SoftThreshold thresh_;
};

} // namespace rl
