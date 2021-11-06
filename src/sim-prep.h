#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

Result
Simple(Parameter const T1p, Parameter const betap, Sequence const seq, long const nRand, Log &log);

} // namespace Sim
