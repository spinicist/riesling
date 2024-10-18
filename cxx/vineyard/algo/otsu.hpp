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
auto OtsuMask(Eigen::ArrayXf const &x, Index const nBins = 128) -> Eigen::ArrayXf;
auto OtsuMask(Eigen::ArrayXf::ConstAlignedMapType const &x, Index const nBins = 128) -> Eigen::ArrayXf;
auto OtsuMasked(Eigen::ArrayXf const &x, Index const nBins = 128) -> Eigen::ArrayXf;
auto OtsuMasked(Eigen::ArrayXf::ConstAlignedMapType const &x, Index const nBins = 128) -> Eigen::ArrayXf;

} // namespace rl