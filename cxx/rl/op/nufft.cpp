#include "nufft.hpp"

#include "../apodize.hpp"
#include "../fft.hpp"
#include "../log/log.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND, typename KF>
NUFFT<ND, KF>::NUFFT(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  : Parent("NUFFT")
  , gridder{opts, traj, nChan, basis}
  , workspace{gridder.ishape}
{
  ishape = Concatenate(traj.matrixForFOV(opts.fov), LastN<2>(gridder.ishape));
  oshape = gridder.oshape;
  std::iota(fftDims.begin(), fftDims.end(), 0);
  Log::Print("NUFFT", "ishape {} oshape {} grid {}", ishape, oshape, gridder.ishape);

  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  for (int ii = 0; ii < 2; ii++) {
    apo_shape[ND + ii] = 1;
    apoBrd_[ND + ii] = ishape[ND + ii];
  }
  apo_ = Apodize<ND, KF>(FirstN<ND>(ishape), FirstN<ND>(gridder.ishape), opts.osamp).reshape(apo_shape); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 0; ii < ND; ii++) {
    padLeft_[ii] = (gridder.ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder.ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND, typename KF>
auto NUFFT<ND, KF>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  -> std::shared_ptr<NUFFT<ND, KF>>
{
  return std::make_shared<NUFFT<ND, KF>>(opts, traj, nChan, basis);
}

template <int ND, typename KF> void NUFFT<ND, KF>::forward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder.forward(workspace, y);
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFT<ND, KF>::adjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) = workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFT<ND, KF>::iforward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder.iforward(workspace, y);
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFT<ND, KF>::iadjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) += workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, true);
}

template struct NUFFT<1>;
template struct NUFFT<2>;
template struct NUFFT<3>;

} // namespace rl::TOps
