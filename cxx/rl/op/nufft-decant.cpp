#include "nufft-decant.hpp"

#include "../fft.hpp"
#include "../log/log.hpp"
#include "apodize.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND, typename KF> NUFFTDecant<ND, KF>::NUFFTDecant(GridOpts<ND> const    &opts,
                                                                TrajectoryN<ND> const &traj,
                                                                CxN<ND + 2> const     &skern,
                                                                Basis::CPtr            basis)
  : Parent("NUFFTDecant")
  , gridder(opts, traj, skern, basis)
  , apo{Concatenate(traj.matrixForFOV(opts.fov), LastN<1>(gridder.ishape)), gridder.ishape, opts.osamp}
  , workspace{gridder.ishape}
{
  ishape = apo.ishape;
  oshape = gridder.oshape;
  std::iota(fftDims.begin(), fftDims.end(), 0);
  Log::Print("NUFFTDecant", "ishape {} oshape {} grid {}", ishape, oshape, gridder.ishape);
}

template <int ND, typename KF> auto
NUFFTDecant<ND, KF>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
  -> std::shared_ptr<NUFFTDecant<ND>>
{
  return std::make_shared<NUFFTDecant<ND>>(opts, traj, skern, basis);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  apo.forward(x, wsm, s);
  FFT::Forward(wsm, fftDims);
  gridder.forward(wscm, y);
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(wsm, fftDims);
  apo.adjoint(wscm, x, s);
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  apo.forward(x, wsm);
  FFT::Forward(wsm, fftDims);
  gridder.iforward(wscm, y, s);
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFTDecant<ND, KF>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  apo.iadjoint(wscm, x);
  this->finishAdjoint(x, time, true);
}

template struct NUFFTDecant<1>;
template struct NUFFTDecant<1, ExpSemi<6>>;
template struct NUFFTDecant<2>;
template struct NUFFTDecant<3>;

} // namespace rl::TOps
