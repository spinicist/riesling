#include "recon-rss.hpp"

namespace rl {

ReconRSSOp::ReconRSSOp(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Operator<3, 3> *sdc,
  std::optional<Re2> basis)
  : Operator<4, 4>()
  , nufft_{make_nufft(traj, ktype, osamp, nC, sdc, basis)}
  , x_{inputDimensions()}
{
}

auto ReconRSSOp::inputDimensions() const -> InputDims { return LastN<4>(nufft_->inputDimensions()); }

auto ReconRSSOp::outputDimensions() const -> OutputDims { return nufft_->outputDimensions(); }

auto ReconRSSOp::forward(Input const &x) const -> Output const &
{
  Log::Fail("ReconRSS has no forward operation");
}

auto ReconRSSOp::adjoint(Output const &y) const -> Input const &
{
  auto const start = Log::Now();
  auto const channels = nufft_->adjoint(y);
  x_ = ConjugateSum(channels, channels).sqrt();
  LOG_DEBUG(FMT_STRING("Finished ReconRSSOp adjoint. Norm {}->{}. Took {}."), Norm(y), Norm(x_), Log::ToNow(start));
  return x_;
}

} // namespace rl