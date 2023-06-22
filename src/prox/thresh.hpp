#pragma once

#include "prox.hpp"

namespace rl::Prox {

struct SoftThreshold final : Prox<Cx> {
    PROX_INHERIT(Cx)
    float λ;

    SoftThreshold(float const λ, Index const sz);
    void apply(float const α, CMap const &x, Map &z) const;
};

} // namespace rl
