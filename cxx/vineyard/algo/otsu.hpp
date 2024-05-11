#pragma once

#include "log.hpp"
#include "tensors.hpp"

namespace rl {

struct OtsuReturn
{
  float thresh;
  Index countAbove;
};

auto Otsu(Eigen::ArrayXf const &x, Index const nBins = 128) -> OtsuReturn;
auto Otsu(Eigen::ArrayXf::ConstMapType const &x, Index const nBins = 128) -> OtsuReturn;

} // namespace rl