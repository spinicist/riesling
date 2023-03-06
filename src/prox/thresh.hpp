#pragma once

#include "prox.hpp"

namespace rl {

template<typename Tensor = Cx4>
struct SoftThreshold final : Prox<Tensor> {
    SoftThreshold(float const λ);
    auto operator()(float const α, Eigen::TensorMap<Tensor const>) const -> Tensor;
private:
    float λ_;
};

} // namespace rl
