#include "otsu.hpp"

namespace rl {

auto Otsu(Eigen::ArrayXf const &x, Index const nBins) -> OtsuReturn
{
  Eigen::ArrayXf::ConstMapType xm(x.data(), x.size());
  return Otsu(xm, nBins);
}

auto Otsu(Eigen::ArrayXf::ConstMapType const &x, Index const nBins) -> OtsuReturn
{
  Index const          n = x.size();
  float const          maxVal = x.maxCoeff();
  Eigen::ArrayXf const thresholds = Eigen::ArrayXf::LinSpaced(nBins, 0.f, maxVal);

  float bestSigma = std::numeric_limits<float>::infinity();
  float bestThresh = 0.f;
  Index bestAbove = 0;
  for (Index ib = 0; ib < nBins; ib++) {
    auto const  mask = x >= thresholds[ib];
    Index const nAbove = mask.count();
    if (nAbove == 0 or nAbove == n) continue;
    float const w1 = nAbove / float(n);
    float const w0 = 1.f - w1;

    Eigen::ArrayXf vals0(n - nAbove), vals1(nAbove);
    Index          ii0 = 0, ii1 = 0;
    for (Index ii = 0; ii < n; ii++) {
      if (mask[ii]) {
        vals1[ii1++] = x[ii];
      } else {
        vals0[ii0++] = x[ii];
      }
    }

    float const var0 = (vals0 - vals0.mean()).square().sum() / (n - nAbove - 1);
    float const var1 = (vals1 - vals1.mean()).square().sum() / (nAbove - 1);

    float const sigma = w0 * var0 + w1 * var1;
    if (sigma < bestSigma) {
      bestSigma = sigma;
      bestThresh = thresholds[ib];
      bestAbove = nAbove;
    }
  }
  Log::Print("Otsu", "Threshold {} retains {}% of voxels", bestThresh, (100.f * bestAbove) / n);
  return OtsuReturn{bestThresh, bestAbove};
}

auto OtsuMask(Eigen::ArrayXf const &x, Index const nBins) -> Eigen::ArrayXf
{
  Eigen::ArrayXf::ConstAlignedMapType xm(x.data(), x.size());
  return OtsuMask(xm, nBins);
}

auto OtsuMask(Eigen::ArrayXf::ConstAlignedMapType const &x, Index const nBins) -> Eigen::ArrayXf
{
  auto const [thresh, count] = Otsu(x, nBins);
  Eigen::ArrayXf masked(count);
  std::copy_if(x.data(), x.data() + x.size(), masked.begin(), [t = thresh](float const f) { return f > t; });
  return masked;
}

auto OtsuMasked(Eigen::ArrayXf const &x, Index const nBins) -> Eigen::ArrayXf
{
  Eigen::ArrayXf::ConstAlignedMapType xm(x.data(), x.size());
  return OtsuMasked(xm, nBins);
}

auto OtsuMasked(Eigen::ArrayXf::ConstAlignedMapType const &x, Index const nBins) -> Eigen::ArrayXf
{
  auto const [thresh, count] = Otsu(x, nBins);
  Eigen::ArrayXf m = (x > thresh).select(Eigen::ArrayXf::Ones(x.size()), Eigen::ArrayXf::Zero(x.size()));
  return m;
}

} // namespace rl