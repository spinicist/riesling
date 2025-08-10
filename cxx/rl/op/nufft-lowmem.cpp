#include "nufft-lowmem.hpp"

#include "../fft.hpp"
#include "../log/log.hpp"
#include "../sense/sense.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
} // namespace

template <int ND, typename KF> NUFFTLowmem<ND, KF>::NUFFTLowmem(GridOpts<ND> const    &opts,
                                                                TrajectoryN<ND> const &traj,
                                                                CxN<ND + 2> const     &sk,
                                                                Basis::CPtr            basis)
  : Parent("NUFFTLowmem")
  , gridder{opts, traj, 1, basis}
  , apo{AddBack(traj.matrixForFOV(opts.fov), gridder.ishape[ND]), FirstN<ND + 1>(gridder.ishape), opts.osamp}
  , nc1{AddFront(LastN<2>(gridder.oshape), 1)}
  , workspace{gridder.ishape}
  , skern{sk}
  , smap{AddBack(FirstN<ND>(gridder.ishape), skern.dimension(DB))}
  , spad{FirstN<ND + 1>(skern.dimensions()), smap.dimensions()}
{
  auto const nB = gridder.ishape[DB];
  auto const nC = skern.dimension(DC);
  ishape = AddBack(traj.matrixForFOV(opts.fov), nB);
  oshape = gridder.oshape;
  oshape[0] = nC;
  std::iota(fftDims.begin(), fftDims.end(), 0);
  Log::Print(this->name, "ishape {} oshape {} grid {} fft {} ws {}", ishape, oshape, gridder.ishape, fftDims,
             workspace.dimensions());

  // Broadcast SENSE across basis if needed
  sbrd.fill(1);
  if (skern.dimension(DB) == 1) {
    sbrd[ND] = nB;
  } else if (skern.dimension(DB) == nB) {
    sbrd[ND] = 1;
  } else {
    throw Log::Failure(this->name, "SENSE kernels had basis dimension {}, expected 1 or {}", skern.dimension(ND), nB);
  }
}

template <int ND, typename KF> auto
NUFFTLowmem<ND, KF>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
  -> std::shared_ptr<NUFFTLowmem<ND, KF>>
{
  return std::make_shared<NUFFTLowmem<ND, KF>>(opts, traj, skern, basis);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::kernToMap(Index const c) const
{
  Log::Print("Lowmem", "Inflating sensitivity for channel {}", c);
  auto const skshape = skern.dimensions();
  auto const sk1 = skern.slice(Sz5{0, 0, 0, 0, c}, AddBack(FirstN<ND + 1>(skshape), 1));
  smap = SENSE::KernelsToMaps(sk1, FirstN<ND>(smap.dimensions()), 1.f, SENSE::Normalization::None);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::forward(InCMap x, OutMap y, float const s) const
{
  auto const     time = this->startForward(x, y, false);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), FirstN<ND + 1>(workspace.dimensions()));
  OutMap         nc1m(nc1.data(), nc1.dimensions());
  ws1m.setZero();
  y.setZero();
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    apo.forward(x, ws1m, s);
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.broadcast(sbrd);
    FFT::Forward(workspace, fftDims);
    gridder.forward(workspace, nc1m);
    y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)}).device(Threads::TensorDevice()) = nc1;
  }
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const     time = this->startForward(x, y, true);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), FirstN<ND + 1>(workspace.dimensions()));
  OutMap         nc1m(nc1.data(), nc1.dimensions());
  ws1m.setZero();
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    apo.forward(x, ws1m, s);
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.broadcast(sbrd);
    FFT::Forward(workspace, fftDims);
    gridder.forward(workspace, nc1m);
    y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)}).device(Threads::TensorDevice()) += nc1;
  }
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const      time = this->startAdjoint(y, x, false);
  CxNMap<ND + 2>  wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1>  ws1m(workspace.data(), FirstN<ND + 1>(workspace.dimensions()));
  CxNCMap<ND + 1> ws1cm(workspace.data(), FirstN<ND + 1>(workspace.dimensions()));
  OutCMap         nc1m(nc1.data(), nc1.dimensions());
  x.setZero();
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    nc1.device(Threads::TensorDevice()) = y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)});
    gridder.adjoint(nc1m, wsm);
    FFT::Adjoint(workspace, fftDims);
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.broadcast(sbrd).conjugate();
    apo.iadjoint(ws1cm, x, s); // This needs to accumulate across channels, so use iadjoint
  }
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const      time = this->startAdjoint(y, x, true);
  CxNMap<ND + 2>  wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1>  ws1m(workspace.data(), FirstN<ND + 1>(workspace.dimensions()));
  CxNCMap<ND + 1> ws1cm(workspace.data(), FirstN<ND + 1>(workspace.dimensions()));
  OutCMap         nc1m(nc1.data(), nc1.dimensions());
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    nc1.device(Threads::TensorDevice()) = y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)});
    gridder.adjoint(nc1m, wsm);
    FFT::Adjoint(workspace, fftDims);
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.broadcast(sbrd).conjugate();
    apo.iadjoint(ws1cm, x, s);
  }
  this->finishAdjoint(x, time, true);
}

template struct NUFFTLowmem<1>;
template struct NUFFTLowmem<2>;
template struct NUFFTLowmem<3>;

} // namespace rl::TOps
