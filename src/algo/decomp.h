#pragma once

#include "log.h"
#include "types.h"

Cx5 LowRankKernels(Cx5 const &m, float const thresh);

struct VecsAndValsCx
{
  Eigen::MatrixXcf vecs;
  Eigen::ArrayXf vals;
};

VecsAndValsCx
PCA(Eigen::Map<Eigen::MatrixXcf const> const &data, Index const nR, float const thresh = -1.f);

struct VecsAndVals
{
  Eigen::MatrixXf vecs;
  Eigen::ArrayXf vals;
};

VecsAndVals SVD(Eigen::ArrayXXf const &mat);