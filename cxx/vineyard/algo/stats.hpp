#pragma once

#include "log.hpp"
#include "types.hpp"

// Statistical Utilities

namespace rl {

auto Covariance(Eigen::Ref<Eigen::MatrixXcf const> const &X, bool const demean = true) -> Eigen::MatrixXcf;
auto Correlation(Eigen::Ref<Eigen::MatrixXcf const> const &X, bool const demean = true) -> Eigen::MatrixXcf;
auto CountBelow(Eigen::Ref<Eigen::ArrayXf const> const &vals, float const thresh) -> Index;
auto Percentiles(Eigen::Ref<Eigen::ArrayXf const> const &vals, std::vector<float> const &p) -> std::vector<float>;

} // namespace rl
