#pragma once

#include "prox.hpp"

namespace rl {

template<typename Tensor>
struct Entropy final : Prox<Tensor> {
    Entropy(float const λ, float const scale);
    auto operator()(float const α, Eigen::TensorMap<Tensor const>) const -> Tensor;
private:
    float λ_, scale_;
};

struct NMREnt final : Prox<Cx4> {
    NMREnt(float const λ, float const scale);
    auto operator()(float const α, Eigen::TensorMap<Cx4 const>) const -> Cx4;
private:
    float λ_, scale_;
};

} // namespace rl
