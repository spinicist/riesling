#pragma once

#include "prox.hpp"

namespace rl {

struct SoftThreshold final : Prox<Cx> {
    PROX_INHERIT(Cx)
    float λ;

    SoftThreshold(float const λ);
    void apply(float const α, CMap const &x, Map &z) const;
};

} // namespace rl
