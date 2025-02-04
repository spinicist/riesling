#pragma once

#include "op/grid.hpp"

namespace rl {

/*
 *  Convenience function for calculating the NUFFT apodization
 */
template <int ND> auto Apodize(Sz<ND> const shape, typename TOps::Grid<ND>::Ptr const g) -> CxN<ND>;

} // namespace rl
