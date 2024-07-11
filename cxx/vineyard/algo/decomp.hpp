#pragma once

#include "types.hpp"

// Wrappers for dynamic decomps so only compile once

namespace rl {

template <typename Scalar = Cx> struct Eig
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eig(Eigen::Ref<Matrix const> const &gramian);
  Matrix         P;
  Eigen::ArrayXf V;
};
extern template struct Eig<float>;
extern template struct Eig<Cx>;

template <typename Scalar = Cx> struct SVD
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  SVD(Eigen::Ref<Matrix const> const &mat);
  Matrix         U, V;
  Eigen::ArrayXf S;

  auto variance(Index const N) const
    -> Eigen::ArrayXf;                           // Calculate the cumulative fractional variance contained in first N vectors
  auto equalized(Index const N) const -> Matrix; // Equalize variance over first N vectors
};
extern template struct SVD<float>;
extern template struct SVD<Cx>;

} // namespace rl
