#pragma once

#include "log.h"
#include "types.h"

// This is the type of the lambda functions to represent the encode/decode operators
// Have not yet updated this to use Operator objects
using EncodeFunction = std::function<void(Cx3 &x, Cx3 &y)>;
using DecodeFunction = std::function<void(Cx3 const &x, Cx3 &y)>;

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
