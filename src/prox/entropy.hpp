#pragma once

#include "prox.hpp"

namespace rl {

struct Entropy final : Prox<Cx> {
    PROX_INHERIT(Cx)
    Entropy(float const λ);
    void apply(float const α, CMap const &x, Map &z) const;
private:
    float λ_;
};

struct NMREntropy final : Prox<Cx> {
    PROX_INHERIT(Cx)
    NMREntropy(float const λ);
    void apply(float const α, CMap const &x, Map &z) const;
private:
    float λ_;
};

} // namespace rl
