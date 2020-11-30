#pragma once

#include "log.h"
#include "types.h"

// Conjugate Gradients with iterative shrinking and thresholding
Cx3 cgisR0(
    Cx3 &radial,
    Cx3::Dimensions const &dims,
    EncodeFunction const &encode,
    DecodeFunction const &decode,
    long const &max_its,
    float const &thresh,
    float const &l1_weight,
    Log &log);

Cx3 cgist(
    Cx3 &radial,
    Cx3::Dimensions const &dims,
    EncodeFunction const &encode,
    DecodeFunction const &decode,
    long const &max_its,
    float const &thresh,
    float const &l1_weight,
    Log &log);
