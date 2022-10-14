#include "recon.hpp"

namespace rl {

ReconOp::ReconOp(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Cx4 const &maps,
  Functor<Cx3> *sdc,
  std::optional<Re2> basis,
  bool const toeplitz)
  : Operator<4, 4>(),
  nufft_{make_nufft(traj, ktype, osamp, maps.dimension(0), LastN<3>(maps.dimensions()), sdc, basis, toeplitz)}
  , sense_{maps, basis ? basis.value().dimension(1) : traj.nFrames()}
{
}

auto ReconOp::inputDimensions() const -> InputDims
{
  return sense_.inputDimensions();
}

auto ReconOp::outputDimensions() const -> OutputDims
{
    return nufft_->outputDimensions();

}

auto ReconOp::forward(Input const &x) const -> Output const &
{
  return nufft_->forward(sense_.forward(x));
}

auto ReconOp::adjoint(Output const &y) const -> Input const &
{
  return sense_.adjoint(nufft_->adjoint(y));
}

auto ReconOp::adjfwd(Input const &x) const -> Input
{
  return sense_.adjoint(nufft_->adjfwd(sense_.forward(x)));
}

} // namespace rl