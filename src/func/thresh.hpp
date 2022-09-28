#pragma once

#include "functor.hpp"

namespace rl {

struct SoftThreshold final : Functor<Cx4> {
    float λ;
    SoftThreshold(float);
    auto operator()(Cx4 const &) const -> Cx4;
};

} // namespace rl
