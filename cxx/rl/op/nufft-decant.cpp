#include "nufft-decant.hpp"

#include "../apodize.hpp"
#include "../fft.hpp"
#include "../log.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND, typename KF>
NUFFTDecant<ND, KF>::NUFFTDecant(GridOpts<ND> const    &opts,
                                 TrajectoryN<ND> const &traj,
                                 CxN<ND + 2> const     &skern,
                                 Basis::CPtr            basis)
  : Parent("NUFFTDecant")
  , gridder(opts, traj, skern, basis)
  , workspace{gridder.ishape}
{
  ishape = Concatenate(FirstN<1>(gridder.ishape), traj.matrixForFOV(opts.fov));
  oshape = gridder.oshape;
  std::iota(fftDims.begin(), fftDims.end(), 1);
  Log::Print("NUFFTDecant", "ishape {} oshape {} grid {}", ishape, oshape, gridder.ishape);

  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  apo_shape[0] = 1;
  apoBrd_[0] = gridder.ishape[0];
  apo_ = Apodize<ND, KF>(LastN<ND>(ishape), LastN<ND>(gridder.ishape), opts.osamp).reshape(apo_shape); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 1; ii < InRank; ii++) {
    padLeft_[ii] = (gridder.ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder.ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND, typename KF>
auto NUFFTDecant<ND, KF>::Make(GridOpts<ND> const    &opts,
                               TrajectoryN<ND> const &traj,
                               CxN<ND + 2> const     &skern,
                               Basis::CPtr            basis) -> std::shared_ptr<NUFFTDecant<ND>>
{
  return std::make_shared<NUFFTDecant<ND>>(opts, traj, skern, basis);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder.forward(workspace, y);
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) = workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder.iforward(workspace, y);
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) += workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, true);
}

template struct NUFFTDecant<1>;
template struct NUFFTDecant<2>;
template struct NUFFTDecant<3>;

} // namespace rl::TOps
