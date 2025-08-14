#pragma once

#include "../types.hpp"

// Wrappers for dynamic decomps so only compile once

namespace rl {

template <typename Scalar = Cx> struct Eig
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RealArray = Eigen::Array<typename Eigen::NumTraits<Scalar>::Real, Eigen::Dynamic, 1>;
  Eig(Eigen::Ref<Matrix const> const &gramian);
  Matrix    P;
  RealArray V;
};

template <typename Scalar = Cx> struct SVD
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RealArray = Eigen::Array<typename Eigen::NumTraits<Scalar>::Real, Eigen::Dynamic, 1>;
  SVD(Eigen::Ref<Matrix const> const &mat);
  Matrix    U, V;
  RealArray S;

  auto variance(Index const N) const -> RealArray; // Calculate the cumulative fractional variance contained in first N vectors
  auto basis(Index const N, bool const scale = false) const -> Matrix;       // Return the first N singular vectors to make a basis
  auto equalized(Index const N) const -> Matrix;   // Equalize variance over first N vectors
};

} // namespace rl
