#include "stats.hpp"

namespace rl {

auto Covariance(Eigen::Ref<Eigen::MatrixXcf const> const &X, bool const demean) -> Eigen::MatrixXcf
{
  if (demean) {
    Eigen::VectorXcf mean = X.rowwise().mean();
    auto const       dm = X.colwise() - mean;
    return (dm.conjugate() * dm.transpose()) / (dm.rows() - 1);
  } else {
    return (X.conjugate() * X.transpose()) / (X.rows() - 1);
  }
}

auto Correlation(Eigen::Ref<Eigen::MatrixXcf const> const &X, bool const demean) -> Eigen::MatrixXcf
{
  auto c = Covariance(X, demean);
  Eigen::ArrayXf σ = c.diagonal().array().real().sqrt();
  c.array().rowwise() /= σ.transpose().cast<Cx>();
  c.array().colwise() /= σ.cast<Cx>();
  return c;
}

auto CountCumulativeBelow(Eigen::Ref<Eigen::ArrayXf const> const &vals, float const thresh) -> Index
{
  Index nRetain = vals.rows();
  if ((thresh > 0.f) && (thresh <= 1.f)) {
    Eigen::ArrayXf cumsum(vals.rows());
    std::partial_sum(vals.begin(), vals.end(), cumsum.begin());
    cumsum /= cumsum.tail(1)(0);
    nRetain = (cumsum < thresh).count();
  }
  return nRetain;
}

auto Percentiles(Eigen::ArrayXf::ConstAlignedMapType vals, std::vector<float> const &ps) -> std::vector<float>
{
  Eigen::ArrayXf x = vals;
  std::sort(x.begin(), x.end());

  std::vector<float> pvals;
  for (auto const p : ps) {
    if (p < 0. || p > 100.) { throw Log::Failure("Stats", "Requested percentile {} outside range 0-100", p); }
    Index const ind = std::clamp(Index(p * (x.size() - 1)), 0L, Index(x.size() - 1));
    pvals.push_back(x[ind]);
  }
  return pvals;
}

auto PercentilesAbove(float const thresh, Eigen::Ref<Eigen::ArrayXf const> const &vals, std::vector<float> const &ps) -> std::vector<float>
{
  Eigen::ArrayXf x = vals;
  std::sort(x.begin(), x.end());

  std::vector<float> pvals;
  for (auto const p : ps) {
    if (p < 0. || p > 100.) { throw Log::Failure("Stats", "Requested percentile {} outside range 0-100", p); }
    Index const ind = std::clamp(Index(p * (x.size() - 1)), 0L, Index(x.size() - 1));
    pvals.push_back(x[ind]);
  }
  return pvals;
}

} // namespace rl
