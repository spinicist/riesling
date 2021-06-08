#pragma once

#include "log.h"
#include "types.h"

Cx5 LowRankKernels(Cx5 const &m, float const thresh, Log const &log);
Cx2 Covariance(Cx2 const &data);
void PCA(Cx2 const &dataIn, Cx2 &vecIn, R1 &valIn, Log const &log);