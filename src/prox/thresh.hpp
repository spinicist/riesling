#pragma once

#include "prox.hpp"

namespace rl {

template<int ND>
struct SoftThreshold final : Prox<Cx> {
    SoftThreshold(float const λ, Sz<ND> const dims);
    void operator()(float const α, Vector const &x, Vector &z) const;
    float λ;
    Sz<ND> shape;
};

} // namespace rl
