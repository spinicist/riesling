#pragma once

#include "../types.cuh"

namespace gw {

auto Preconditioner(DTensor<TDev, 3> const &T, int const nI, int const nJ, int const nK) -> DTensor<TDev, 2>;

}