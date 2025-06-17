#include "nufft.hpp"

#include "../fft.hpp"
#include "../log/log.hpp"
#include "apodize.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND, typename KF>
NUFFT<ND, KF>::NUFFT(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  : Parent("NUFFT")
  , gridder{opts, traj, nChan, basis}
  , apo{Concatenate(traj.matrixForFOV(opts.fov), LastN<2>(gridder.ishape)), gridder.ishape, opts.osamp}
  , workspace{gridder.ishape}
{
  ishape = apo.ishape;
  oshape = gridder.oshape;
  std::iota(fftDims.begin(), fftDims.end(), 0);
  Log::Print("NUFFT", "ishape {} oshape {} grid {}", ishape, oshape, gridder.ishape);
}

template <int ND, typename KF>
auto NUFFT<ND, KF>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  -> TOp<Cx, ND + 2, 3>::Ptr
{
  return std::make_shared<NUFFT<ND, KF>>(opts, traj, nChan, basis);
}

template <int ND, typename KF> void NUFFT<ND, KF>::forward(InCMap x, OutMap y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  apo.forward(x, wsm);
  FFT::Forward(wsm, fftDims);
  gridder.forward(wscm, y);
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFT<ND, KF>::adjoint(OutCMap y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(wsm, fftDims);
  apo.adjoint(wscm, x);
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFT<ND, KF>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  apo.forward(x, wsm);
  FFT::Forward(wsm, fftDims);
  gridder.iforward(wscm, y, s);
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFT<ND, KF>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  apo.iadjoint(wscm, x);
  this->finishAdjoint(x, time, true);
}

template struct NUFFT<1, rl::ExpSemi<4>>;
template struct NUFFT<2, rl::ExpSemi<4>>;
template struct NUFFT<3, rl::ExpSemi<4>>;

template struct NUFFT<1, rl::TopHat<1>>;
template struct NUFFT<2, rl::TopHat<1>>;
template struct NUFFT<3, rl::TopHat<1>>;

} // namespace rl::TOps
