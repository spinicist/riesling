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
  Index const randN,
  Log &log);

Result T1Prep(Parameter const T1, Sequence const seq, Index const randN, Log &log);

Result
T2Prep(Parameter const T1, Parameter const T2, Sequence const seq, Index const randN, Log &log);

Result
T1T2Prep(Parameter const T1, Parameter const T2, Sequence const seq, Index const randN, Log &log);

} // namespace Sim
