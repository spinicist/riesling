#pragma once

#include "prox.hpp"

namespace rl {

struct Entropy final : Prox<Cx> {
    Entropy(float const λ);
    void operator()(float const α, Vector const &x, Vector &z) const;
private:
    float λ_;
};

struct NMREntropy final : Prox<Cx> {
    NMREntropy(float const λ);
    void operator()(float const α, Vector const &x, Vector &z) const;
private:
    float λ_;
};

} // namespace rl
