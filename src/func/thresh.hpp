#pragma once

#include "functor.hpp"

namespace rl {

struct SoftThreshold final : Prox<Cx4> {
    SoftThreshold();
    auto operator()(float const λ, Eigen::TensorMap<Cx4 const>) const -> Eigen::TensorMap<Cx4>;
};

} // namespace rl
