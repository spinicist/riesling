#pragma once

#include "log.h"
#include "types.h"

Cx5 LowRankKernels(Cx5 const &m, float const thresh);

struct PrincipalComponents
{
  Eigen::MatrixXcf vecs;
  Eigen::VectorXf vals;
};

PrincipalComponents
PCA(Eigen::Map<Eigen::MatrixXcf const> const &data, Index const nR, float const thresh = -1.f);
