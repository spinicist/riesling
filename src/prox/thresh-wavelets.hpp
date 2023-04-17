#pragma once

#include "thresh.hpp"
#include "op/wavelets.hpp"
#include "op/pad.hpp"

namespace rl {

struct ThresholdWavelets final : Prox<Cx> {
    ThresholdWavelets(float const λ, Sz4 const shape, Index const width, Index const levels);
    void operator()(float const α, Vector const &x, Vector &z) const;

private:
    std::shared_ptr<Op::Operator<Cx>> waves_;
    SoftThreshold<4> thresh_;
};

} // namespace rl
