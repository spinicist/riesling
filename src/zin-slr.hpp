#pragma once

#include "op/fft.hpp"
#include "types.h"

namespace rl {
Cx5 zinSLR(Cx5 const &channels, FFTOp<5> const &fft, Index const kSz, float const thresh);
}
