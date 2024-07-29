#pragma once

#include "kernel/kernel.hpp"

namespace rl {

/*
 *  Convenience function for calculating the NUFFT apodization
 */
template <int N> auto Apodize(Sz<N> const shape, Sz<N> const gshape, std::shared_ptr<KernelBase<Cx, N>> const &k) -> CxN<N>;

} // namespace rl
