#pragma once

#include "op/grid.hpp"

namespace rl {

/*
 *  Convenience function for calculating the NUFFT apodization
 */
template <int ND, typename KType> auto Apodize(Sz<ND> const shape, Sz<ND> const gshape, float const osamp) -> CxN<ND>;

} // namespace rl
