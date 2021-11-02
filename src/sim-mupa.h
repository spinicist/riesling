#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

Result MUPA(
    Parameter const T1,
    Parameter const T2,
    Parameter const B1,
    Sequence const seq,
    long const randN,
    Log &log);

}
