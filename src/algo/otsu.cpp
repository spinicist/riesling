#include "otsu.hpp"

namespace rl {

auto Otsu(Eigen::Map<Eigen::ArrayXf const> const &x, Index const nBins) -> OtsuReturn
{
  Index const n = x.size();
  float const maxVal = x.maxCoeff();
  Eigen::ArrayXf const thresholds = Eigen::ArrayXf::LinSpaced(nBins, 0.f, maxVal);

  float bestSigma = std::numeric_limits<float>::infinity();
  float bestThresh = 0.f;
  Index bestAbove = 0;
  for (Index ib = 0; ib < nBins; ib++) {
    auto const mask = x >= thresholds[ib];
    Index const nAbove = mask.count();
    if (nAbove == 0 or nAbove == n)
      continue;
    float const w1 = nAbove / float(n);
    float const w0 = 1.f - w1;

    Eigen::ArrayXf vals0(n - nAbove), vals1(nAbove);
    Index ii0 = 0, ii1 = 0;
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
  return OtsuReturn{bestThresh, bestAbove};
}

} // namespace rl