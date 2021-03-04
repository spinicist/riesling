#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include <math.h>

Cx3 SheppLoganPhantom(
    Info const &info,
    Eigen::Vector3f const &center,
    float const radius,
    float const intensity,
    Log const &log);
