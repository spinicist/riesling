#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

SimResult Simple(
    long const nT1,
    float const T1Lo,
    float const T1Hi,
    long const nbeta,
    float const betaLo,
    float const betaHi,
    bool const betaLog,
    Sequence const seq,
    Log &log);

SimResult PhaseCycled(
    long const nT1,
    float const T1Lo,
    float const T1Hi,
    long const nbeta,
    float const betaLo,
    float const betaHi,
    Sequence const seq,
    Log &log);

} // namespace Sim
