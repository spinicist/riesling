#pragma once

#include "prox.hpp"
#include "op/ops.hpp"

namespace rl::Prox {

struct L2 final : Prox<Cx> {
    PROX_INHERIT(Cx)
    float λ;
    CMap y;

    L2(float const λ, CMap const bias);
    void apply(float const α, CMap const &x, Map &z) const;
    void apply(std::shared_ptr<Ops::Op<Cx>> const α, CMap const &x, Map &z) const;
};

}