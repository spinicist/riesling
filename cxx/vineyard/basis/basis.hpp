#pragma once

#include "types.hpp"

namespace rl {

using Basis = Cx3;

auto IdBasis() -> Basis;
auto ReadBasis(std::string const &basisFile) -> Basis;

} // namespace rl
