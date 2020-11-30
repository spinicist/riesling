#pragma once

#include "log.h"
#include "types.h"

// Nonlinear Conjugate Gradient
Cx3 nlcg(
    Cx3 &radial,
    Cx3::Dimensions const &dims,
    EncodeFunction const &encode,
    DecodeFunction const &decode,
    long const &max_its,
    float const &thresh,
    float const &lambda,
    Log &log);
