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
  auto σ = c.diagonal().array();
  c.array().rowwise() /= σ.transpose();
  c.array().colwise() /= σ;
  return c;
}

auto Threshold(Eigen::Ref<Eigen::ArrayXf const> const &vals, float const thresh) -> Index
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

} // namespace rl
