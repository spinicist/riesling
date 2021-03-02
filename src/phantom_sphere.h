#pragma once

#include "info.h"
#include "log.h"
#include "types.h"

Cx3 SphericalPhantom(
    Info const &info,
    Eigen::Vector3f const &center,
    float const radius,
    float const intensity,
    Log const &log);
