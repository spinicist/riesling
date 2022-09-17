#include "parameter.hpp"

#include "../log.hpp"
#include <random>

namespace rl {

Parameter const T1wm{1.0, 0.15, 0.5, 2.0, true}, T1gm{1.5, 0.25, 0.5, 3.5, true}, T1csf{3.5, 0.5, 2., 6., true};
Parameter const T2wm{0.08, 0.03, 0.01, 0.2, true}, T2gm{0.12, 0.05, 0.01, 1, true},
  T2csf{1.5, 0.5, 1, 2};
Parameter const B1{1.0, 0.5, 0.25, 1.5, true};

Tissue::Tissue(std::vector<Parameter> const pars)
  : means_(pars.size())
  , stds_(pars.size())
  , los_(pars.size())
  , his_(pars.size())
  , uni_(pars.size())
{
  for (size_t ii = 0; ii < pars.size(); ii++) {
    means_[ii] = pars[ii].mean;
    stds_[ii] = pars[ii].std;
    los_[ii] = pars[ii].lo;
    his_[ii] = pars[ii].hi;
    uni_[ii] = pars[ii].uniform;
  }
}

Index Tissue::nP() const
{
  return means_.size();
}

Eigen::ArrayXXf Tissue::values(Index const NV) const
{
  std::random_device rd;
  std::mt19937 gen(rd());
  Eigen::ArrayXXf vals(means_.rows(), NV);
  for (Index ii = 0; ii < means_.rows(); ii++) {
    if (uni_[ii]) {
      std::uniform_real_distribution<float> dis(los_[ii], his_[ii]);
      vals.row(ii) = vals.row(ii).unaryExpr([&](float dummy) { return dis(gen); });
    } else {
      std::normal_distribution<float> dis(means_[ii], stds_[ii]);
      vals.row(ii) = vals.row(ii).unaryExpr(
        [&](float dummy) { return std::clamp(dis(gen), los_[ii], his_[ii]); });
    }
  }
  return vals;
}

Tissues::Tissues(std::vector<Tissue> const tissues)
  : tissues_{tissues}
{
  for (size_t ii = 1; ii < tissues_.size(); ii++) {
    if (tissues_[ii].nP() != tissues_[0].nP()) {
      Log::Fail("Tissues had different numbers of parameters");
    }
  }
  nP_ = tissues_[0].nP();
}

Index Tissues::nP() const
{
  return nP_;
}

Eigen::ArrayXXf Tissues::values(Index const nsamp) const
{
  Eigen::ArrayXXf parameters(nP_, tissues_.size() * nsamp);

  for (size_t ii = 0; ii < tissues_.size(); ii++) {
    parameters.middleCols(ii * nsamp, nsamp) = tissues_[ii].values(nsamp);
  }

  return parameters;
}

} // namespace rl
