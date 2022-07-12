#pragma once

#include "types.h"

namespace rl {
void ImageTukey(float const &start_r, float const &end_r, float const &end_h, Cx3 &x);
void KSTukey(float const &start_r, float const &end_r, float const &end_h, Cx4 &x);
} // namespace rl
