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
    long const ngamma,
    float const betaLo,
    float const betaHi,
    long const sps,
    float const alpha,
    float const TR,
    float const TI,
    float const Trec,
    Log &log);

}
