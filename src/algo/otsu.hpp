#pragma once

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

struct OtsuReturn {
    float thresh;
    Index countAbove;
};

auto Otsu(Eigen::Map<Eigen::ArrayXf const> const &x, Index const nBins = 128) -> OtsuReturn;

} // namespace rl