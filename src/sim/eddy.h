#pragma once

#include "log.h"
#include "sim.h"
#include "types.h"

namespace Sim {

Result Eddy(
  Parameter const T1,
  Parameter const beta,
  Parameter const gamma,
  Sequence const seq,
  Index const nRand);

}
