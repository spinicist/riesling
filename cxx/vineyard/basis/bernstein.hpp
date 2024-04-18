#pragma once

#include "basis.hpp"

namespace rl {

auto BernsteinPolynomial(Index const N, Index const traces) -> Basis<Cx>;

} // namespace rl
