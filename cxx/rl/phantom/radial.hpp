#pragma once

#include "../types.hpp"

namespace rl {
Re3 ArchimedeanSpiral(Index const matrix, float const OS, Index const traces);
Re3 Phyllotaxis(Index const matrix, float const OS, Index const spk, Index const smooth, Index const spi);
} // namespace rl
