#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

Result Simple(
  Parameter const T1p,
  Parameter const betap,
  Parameter const B1p,
  Sequence const seq,
  Log &log);

} // namespace Sim
