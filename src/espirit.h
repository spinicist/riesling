#pragma once

#include "info.h"
#include "kernel.h"
#include "log.h"

Cx4 ESPIRIT(
    Info const &info,
    R3 const &traj,
    float const os,
    Kernel *const kb,
    long const calSz,
    long const kernelSz,
    float const retain,
    Cx3 const &data,
    Log &log);
