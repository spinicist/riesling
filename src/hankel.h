#pragma once

#include "log.h"
#include "types.h"

Cx5 ToKernels(Cx4 const &grid, Index const kRad, Index const calRad, Index const gapRad);
void FromKernels(Index const blkSz, Index const kSz, Cx2 const &kernels, Cx4 &grid);
