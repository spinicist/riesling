#pragma once

#include "info.hpp"

namespace rl {
Re3 ArchimedeanSpiral(Index const matrix, float const OS, Index const traces);
Re3 Phyllotaxis(Index const matrix, float const OS, Index const spk, Index const smooth, Index const spi, bool const gm);
} // namespace rl
