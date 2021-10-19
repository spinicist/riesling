#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

SimResult Diffusion(
    long const nT1,
    float const T1Lo,
    float const T1Hi,
    long const nbeta,
    float const betaLo,
    float const betaHi,
    long const ngamma,
    Sequence const seq,
    Log &log);

}
