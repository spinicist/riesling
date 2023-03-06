#pragma once

#include "prox.hpp"

namespace rl {

struct Entropy final : Prox<Cx4> {
    Entropy(float const λ);
    auto operator()(float const α, Eigen::TensorMap<Cx4 const>) const -> Cx4;
private:
    float λ_;
};

struct NMREnt final : Prox<Cx4> {
    NMREnt(float const λ);
    auto operator()(float const α, Eigen::TensorMap<Cx4 const>) const -> Cx4;
private:
    float λ_;
};

} // namespace rl
