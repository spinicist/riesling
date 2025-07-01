#pragma once

#include "types.hpp"

namespace rl {
void NoncartesianTukey(float const &start_r, float const &end_r, float const &end_h, Re3 const &coords, Cx4 &x);
void NoncartesianTukey(float const &start_r, float const &end_r, float const &end_h, Re3 const &coords, Cx5 &x);
} // namespace rl
