#include "nufft-lowmem.hpp"

#include "../apodize.hpp"
#include "../fft.hpp"
#include "../log/log.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
template <int NDp2> auto OneChannel(Sz<NDp2> shape) -> Sz<NDp2 - 1>
{
  Sz<NDp2 - 1> out;
  std::copy_n(shape.begin(), NDp2 - 2, out.begin());
  out[NDp2 - 2] = 1;
  return out;
}

template <int NDp2> auto NoChannels(Sz<NDp2> shape) -> Sz<NDp2 - 1>
{
  Sz<NDp2 - 1> out;
  std::copy_n(shape.begin(), NDp2 - 2, out.begin());
  out[NDp2 - 2] = shape[NDp2 - 1];
  return out;
}
} // namespace

template <int ND, typename KF>
NUFFTLowmem<ND, KF>::NUFFTLowmem(GridOpts<ND> const    &opts,
                                 TrajectoryN<ND> const &traj,
                                 CxN<ND + 2> const     &sk,
                                 Basis::CPtr            basis)
  : Parent("NUFFTLowmem")
  , gridder{Grid<ND, KF>::Make(opts, traj, 1, basis)}
  , nc1{AddFront(LastN<2>(gridder->oshape), 1)}
  , workspace{gridder->ishape}
  , skern{sk}
  , smap{AddBack(FirstN<ND>(gridder->ishape), skern.dimension(DB))}
  , spad{OneChannel(skern.dimensions()), smap.dimensions()}
{
  auto const nB = gridder->ishape[DB];
  auto const nC = skern.dimension(DC);
  ishape = AddBack(traj.matrixForFOV(opts.fov), nB);
  oshape = gridder->oshape;
  oshape[0] = nC;
  std::iota(fftDims.begin(), fftDims.end(), 0);
  Log::Print(this->name, "ishape {} oshape {} grid {} fft {} ws {}", ishape, oshape, gridder->ishape, fftDims, workspace.dimensions());

  // Broadcast SENSE across basis if needed
  sbrd.fill(1);
  if (skern.dimension(ND + 1) == 1) {
    sbrd[ND] = nB;
  } else if (skern.dimension(ND + 1) == nB) {
    sbrd[ND] = 1;
  } else {
    throw Log::Failure(this->name, "SENSE kernels had basis dimension {}, expected 1 or {}", skern.dimension(ND + 2), nB);
  }
  // Calculate apodization correction
  auto apo_shape = FirstN<ND>(ishape);
  apoBrd_.fill(1);
  apoBrd_[ND] = nB;
  apo_ = Apodize<ND, KF>(apo_shape, FirstN<ND>(gridder->ishape), opts.osamp).reshape(AddBack(apo_shape, 1)); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 0; ii < ND; ii++) {
    padLeft_[ii] = (gridder->ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder->ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND, typename KF>
auto NUFFTLowmem<ND, KF>::Make(GridOpts<ND> const    &opts,
                               TrajectoryN<ND> const &traj,
                               CxN<ND + 2> const     &skern,
                               Basis::CPtr            basis) -> std::shared_ptr<NUFFTLowmem<ND, KF>>
{
  return std::make_shared<NUFFTLowmem<ND, KF>>(opts, traj, skern, basis);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::kernToMap(Index const c) const
{
  float const scale = std::sqrt(Product(FirstN<ND>(smap.dimensions())) / (float)Product(FirstN<ND>(skern.dimensions())));
  smap.setZero();
  CxN<ND + 1> const     sk1 = skern.template chip<DC>(c) * Cx(scale);
  CxNCMap<ND + 1> const sk1map(sk1.data(), sk1.dimensions());
  CxNMap<ND + 1>        smapmap(smap.data(), smap.dimensions());
  spad.forward(sk1map, smapmap);
  Sz<ND> ftd;
  std::iota(ftd.begin(), ftd.end(), 0);
  FFT::Adjoint(smap, ftd);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::forward(InCMap x, OutMap y) const
{
  auto const     time = this->startForward(x, y, false);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), AddBack(FirstN<ND>(workspace.dimensions()), workspace.dimension(ND + 1)));
  OutMap         nc1m(nc1.data(), nc1.dimensions());
  ws1m.setZero();
  y.setZero();
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    ws1m.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_) * smap.broadcast(sbrd);
    FFT::Forward(workspace, fftDims);
    gridder->forward(workspace, nc1m);
    y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)}).device(Threads::TensorDevice()) = nc1;
  }
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::iforward(InCMap x, OutMap y) const
{
  auto const     time = this->startForward(x, y, true);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), AddBack(FirstN<ND>(workspace.dimensions()), workspace.dimension(ND + 1)));
  OutMap         nc1m(nc1.data(), nc1.dimensions());
  ws1m.setZero();
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    ws1m.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_) * smap.broadcast(sbrd);
    FFT::Forward(workspace, fftDims);
    gridder->forward(workspace, nc1m);
    y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)}).device(Threads::TensorDevice()) += nc1;
  }
  this->finishForward(y, time, true);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::adjoint(OutCMap y, InMap x) const
{
  auto const     time = this->startAdjoint(y, x, false);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  OutCMap        nc1m(nc1.data(), nc1.dimensions());
  x.setZero();
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    nc1.device(Threads::TensorDevice()) = y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)});
    gridder->adjoint(nc1m, wsm);
    FFT::Adjoint(workspace, fftDims);
    CxNMap<ND + 1> ws1m(workspace.data(), NoChannels(workspace.dimensions()));
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.conjugate().broadcast(sbrd);
    x.device(Threads::TensorDevice()) += ws1m.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  }
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void NUFFTLowmem<ND, KF>::iadjoint(OutCMap y, InMap x) const
{
  auto const     time = this->startAdjoint(y, x, true);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), NoChannels(workspace.dimensions()));
  OutCMap        nc1m(nc1.data(), nc1.dimensions());
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    nc1.device(Threads::TensorDevice()) = y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)});
    gridder->adjoint(nc1m, wsm);
    FFT::Adjoint(workspace, fftDims);
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.conjugate().broadcast(sbrd);
    x.device(Threads::TensorDevice()) += ws1m.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  }
  this->finishAdjoint(x, time, true);
}

template struct NUFFTLowmem<1>;
template struct NUFFTLowmem<2>;
template struct NUFFTLowmem<3>;

} // namespace rl::TOps
