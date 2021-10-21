#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

Result Eddy(
  Parameter const T1,
  Parameter const beta,
  Parameter const gamma,
  Parameter const B1,
  bool const negPC,
  Sequence const seq,
  Log &log);

}
