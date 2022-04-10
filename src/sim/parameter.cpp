#include "parameter.h"

#include <random>

namespace Sim {

Tissue::Tissue(std::vector<Parameter> const &pars)
  : means(pars.size())
  , stds(pars.size())
  , los(pars.size())
  , his(pars.size())
{
  for (size_t ii = 0; ii < pars.size(); ii++) {
    means[ii] = pars[ii].mean;
    stds[ii] = pars[ii].std;
    los[ii] = pars[ii].lo;
    his[ii] = pars[ii].hi;
  }
}

Eigen::ArrayXXf Tissue::values(Index const NV)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  Eigen::ArrayXXf vals(means.rows(), NV);
  for (Index ii = 0; ii < means.rows(); ii++) {
    std::normal_distribution<float> dis(means[ii], stds[ii]);
    vals.row(ii) =
      vals.row(ii).unaryExpr([&](float dummy) { return std::clamp(dis(gen), los[ii], his[ii]); });
  }
  return vals;
}

} // namespace Sim
