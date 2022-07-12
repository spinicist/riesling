#pragma once

#include "log.h"
#include "types.h"

namespace rl {
struct PCAResult
{
  Eigen::MatrixXcf vecs;
  Eigen::ArrayXf vals;
};

PCAResult PCA(Eigen::Map<Eigen::MatrixXcf const> const &data, Index const nR, float const thresh = -1.f);

// Wrapper for dynamic sized SVDs so it only gets compiled once
template <typename Scalar>
struct SVD
{
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  SVD(Eigen::Ref<Matrix const> const &mat, bool const transpose = false, bool const verbose = false);
  Matrix U, V;
  Eigen::ArrayXf vals;
};

extern template struct SVD<float>;
extern template struct SVD<Cx>;
}
