#pragma once

#include "types.hpp"

namespace rl {

template<typename Scalar, int N>
auto Pad(Eigen::Tensor<Scalar, N> const &t, Sz<N> const oshape) -> Eigen::Tensor<Scalar, N>;

template<typename Scalar, int N>
auto Crop(Eigen::Tensor<Scalar, N> const &t, Sz<N> const oshape) -> Eigen::Tensor<Scalar, N>;

}