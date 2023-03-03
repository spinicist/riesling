#pragma once

#include "functor.hpp"

namespace rl {

struct ProxEnt final : Prox<Cx4> {
    ProxEnt(float const λ);
    auto operator()(float const α, Eigen::TensorMap<Cx4 const>) const -> Cx4;
private:
    float λ_;
};

} // namespace rl
