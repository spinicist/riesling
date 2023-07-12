#pragma once

#include "log.hpp"
#include "types.hpp"

// Wrappers for dynamic decomps so only compile once

namespace rl {

template <typename Scalar = Cx>
struct Eig
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eig(Eigen::Ref<Matrix const> const &gramian);
  Matrix         P;
  Eigen::ArrayXf V;
};
extern template struct Eig<float>;
extern template struct Eig<Cx>;

template <typename Scalar = Cx>
struct SVD
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  SVD(Eigen::Ref<Matrix const> const &mat);
  Matrix         U, V;
  Eigen::ArrayXf S;
};
extern template struct SVD<float>;
extern template struct SVD<Cx>;

} // namespace rl
