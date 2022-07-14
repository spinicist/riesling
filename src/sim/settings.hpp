#pragma once

#include "types.h"

namespace rl {

struct Settings
{
  Index sps, gps, gprep2;
  float alpha, ascale, TR, Tramp, Tssi, TI, TI2, Trec, TE, bval;
};

} // namespace rl
