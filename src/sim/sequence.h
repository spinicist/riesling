#pragma once

#include "types.h"

namespace Sim {

struct Sequence
{
  Index sps, gps;
  float alpha, TR, Tramp, Tssi, TI, Trec, TE;
};

} // namespace Sim
