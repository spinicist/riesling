#pragma once

#include "gridder.h"
#include "log.h"

Cx4 ESPIRIT(
    Gridder const &hires,
    Gridder const &lores,
    Cx3 const &data,
    long const kernelRad,
    long const calRad,
    Log &log);
