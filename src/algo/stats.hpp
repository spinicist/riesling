#pragma once

#include "log.hpp"
#include "types.hpp"

// Statistical Utilities

namespace rl {

auto Covariance(Eigen::Ref<Eigen::MatrixXcf const> const &X) -> Eigen::MatrixXcf;
auto Correlation(Eigen::Ref<Eigen::MatrixXcf const> const &X) -> Eigen::MatrixXcf;
auto Threshold(Eigen::Ref<Eigen::ArrayXf const> const &vals, float const thresh) -> Index;

} // namespace rl
