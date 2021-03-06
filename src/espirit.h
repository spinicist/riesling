#pragma once

#include "gridder.h"
#include "log.h"

Cx4 ESPIRIT(
    Gridder const &grid,
    Cx3 const &data,
    long const kernelRad,
    long const calRad,
    float const thresh,
    Log &log);
