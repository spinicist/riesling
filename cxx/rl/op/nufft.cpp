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

template <int ND, typename KF> void NUFFT<ND, KF>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  apo.forward(x, wsm, s);
  FFT::Forward(wsm, fftDims);
  gridder.forward(wscm, y);
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFT<ND, KF>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(wsm, fftDims);
  apo.adjoint(wscm, x, s);
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFT<ND, KF>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  apo.forward(x, wsm, s);
  FFT::Forward(wsm, fftDims);
  gridder.iforward(wscm, y);
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFT<ND, KF>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder.ishape);
  InCMap     wscm(workspace.data(), gridder.ishape);
  gridder.adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  apo.iadjoint(wscm, x, s);
  this->finishAdjoint(x, time, true);
}

template struct NUFFT<1, rl::ExpSemi<4>>;
template struct NUFFT<2, rl::ExpSemi<4>>;
template struct NUFFT<3, rl::ExpSemi<4>>;

template struct NUFFT<1, rl::ExpSemi<6>>;
template struct NUFFT<2, rl::ExpSemi<6>>;
template struct NUFFT<3, rl::ExpSemi<6>>;

template struct NUFFT<1, rl::ExpSemi<8>>;
template struct NUFFT<2, rl::ExpSemi<8>>;
template struct NUFFT<3, rl::ExpSemi<8>>;

template struct NUFFT<1, rl::TopHat<1>>;
template struct NUFFT<2, rl::TopHat<1>>;
template struct NUFFT<3, rl::TopHat<1>>;

template <int ND> auto MakeNUFFT(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  -> TOp<ND + 2, 3>::Ptr
{
  typename TOps::TOp<ND + 2, 3>::Ptr nufft = nullptr;
  if (opts.tophat) {
    nufft = std::make_shared<TOps::NUFFT<ND, TopHat<1>>>(opts, traj, nChan, basis);
  } else {
    switch (opts.kW) {
      case 4: nufft = std::make_shared<TOps::NUFFT<ND, ExpSemi<4>>>(opts, traj, nChan, basis); break;
      case 6: nufft = std::make_shared<TOps::NUFFT<ND, ExpSemi<6>>>(opts, traj, nChan, basis); break;
      case 8: nufft = std::make_shared<TOps::NUFFT<ND, ExpSemi<8>>>(opts, traj, nChan, basis); break;
      default:
        throw(Log::Failure("NUFFT", "Kernel width {} not supported", opts.kW));
    }
  }
  return nufft;
}

template auto MakeNUFFT<1>(GridOpts<1> const &, TrajectoryN<1> const &, Index const, Basis::CPtr) -> TOp<3, 3>::Ptr;
template auto MakeNUFFT<2>(GridOpts<2> const &, TrajectoryN<2> const &, Index const, Basis::CPtr) -> TOp<4, 3>::Ptr;
template auto MakeNUFFT<3>(GridOpts<3> const &, TrajectoryN<3> const &, Index const, Basis::CPtr) -> TOp<5, 3>::Ptr;

} // namespace rl::TOps
