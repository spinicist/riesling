#pragma once

#include "log.h"
#include "types.h"

Cx5 LowRankKernels(Cx5 const &m, float const thresh);

struct PCAResult
{
  Eigen::MatrixXcf vecs;
  Eigen::ArrayXf vals;
};

PCAResult
PCA(Eigen::Map<Eigen::MatrixXcf const> const &data, Index const nR, float const thresh = -1.f);

struct SVDResult
{
  Eigen::MatrixXf v, u;
  Eigen::ArrayXf vals;
};

SVDResult SVD(Eigen::ArrayXXf const &mat);