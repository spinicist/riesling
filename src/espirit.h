#pragma once

#include "log.h"
#include "op/grid.h"

Cx4 ESPIRIT(
    std::unique_ptr<GridOp> const &grid,
    Cx3 const &data,
    long const kernelRad,
    long const calRad,
    long const gap,
    float const thresh,
    Log &log);
