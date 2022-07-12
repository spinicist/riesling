#pragma once

#include "info.h"

namespace rl {
R3 ArchimedeanSpiral(Index const read, Index const spokes);
R3 Phyllotaxis(Index const rd, Index const spk, Index const smooth, Index const spi, bool const gm);
} // namespace rl
