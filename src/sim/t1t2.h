#pragma once

#include "log.h"
#include "sequence.h"
#include "types.h"

namespace Sim {

Eigen::ArrayXf T1T2Prep(Sequence const &seq, float const T1, float const T2);

} // namespace Sim
