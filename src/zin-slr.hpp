#pragma once

#include "types.h"
#include "op/fft.hpp"

Cx5 zinSLR(Cx5 const &channels, FFTOp<5> const &fft, Index const kSz, float const thresh);
