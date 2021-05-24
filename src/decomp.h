#pragma once

#include "log.h"
#include "types.h"

Cx5 LowRankKernels(Cx5 const &m, float const thresh, Log const &log);
Cx2 Covariance(Cx2 const &data);
void PCA(Cx2 const &gram, Cx2 &vecs, R1 &vals, Log const &log);