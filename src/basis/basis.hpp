#pragma once

#include "types.hpp"

namespace rl {

template<typename Scalar = Cx>
using Basis = Eigen::Tensor<Scalar, 2>;

template<typename Scalar = Cx>
auto IdBasis() -> Basis<Scalar>;

auto ReadBasis(std::string const &basisFile) -> Basis<Cx>;

} // namespace rl
