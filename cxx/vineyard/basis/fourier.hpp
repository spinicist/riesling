#pragma once

#include "basis.hpp"

namespace rl {

struct FourierBasis
{
  FourierBasis(Index const N, Index const samples, Index const traces, float const os);
  Cx3 basis;
};

} // namespace rl
