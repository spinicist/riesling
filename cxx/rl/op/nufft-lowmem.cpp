#include "nufft-lowmem.hpp"

#include "../apodize.hpp"
#include "../fft.hpp"
#include "../log.hpp"
#include "top-impl.hpp"

#include <numbers>
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;

namespace rl::TOps {

namespace {
template <int ND> auto OneChannel(Sz<ND> shape) -> Sz<ND>
{
  shape[1] = 1;
  return shape;
}

template <int ND> auto NoChannels(Sz<ND> shape) -> Sz<ND - 1>
{
  Sz<ND - 1> out;
  out[0] = shape[0];
  std::copy_n(shape.begin() + 2, ND - 2, out.begin() + 1);
  return out;
}
} // namespace

template <int ND>
NUFFTLowmem<ND>::NUFFTLowmem(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &sk, Basis::CPtr basis)
  : Parent("NUFFTLowmem")
  , gridder{GType::Make(opts, traj, 1, basis)}
  , nc1{AddFront(LastN<2>(gridder->oshape), 1)}
  , workspace{gridder->ishape}
  , skern{sk}
  , smap{Concatenate(FirstN<1>(skern.dimensions()), LastN<ND>(gridder->ishape))}
  , spad{NoChannels(skern.dimensions()), smap.dimensions()}
{
  ishape = Concatenate(FirstN<1>(gridder->ishape), traj.matrixForFOV(opts.fov));
  oshape = gridder->oshape;
  oshape[0] = skern.dimension(1);
  std::iota(fftDims.begin(), fftDims.end(), 2);
  Log::Print("NUFFTLowmem", "ishape {} oshape {} grid {}", ishape, oshape, gridder->ishape);

  // Broadcast SENSE across basis if needed
  sbrd.fill(1);
  if (skern.dimension(0) == 1) {
    sbrd[0] = ishape[0];
  } else if (skern.dimension(0) == ishape[0]) {
    sbrd[0] = 1;
  } else {
    throw Log::Failure("TOp", "SENSE kernels had basis dimension {}, expected 1 or {}", skern.dimension(0), ishape[0]);
  }
  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  apo_shape[0] = 1;
  apoBrd_[0] = gridder->ishape[0];
  apo_ = Apodize(LastN<ND>(ishape), LastN<ND>(gridder->ishape), gridder->kernel).reshape(apo_shape); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 1; ii < InRank; ii++) {
    padLeft_[ii] = (gridder->ishape[ii + 1] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder->ishape[ii + 1] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND>
auto NUFFTLowmem<ND>::Make(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr basis)
  -> std::shared_ptr<NUFFTLowmem<ND>>
{
  return std::make_shared<NUFFTLowmem<ND>>(opts, traj, skern, basis);
}

template <int ND> void NUFFTLowmem<ND>::kernToMap(Index const c) const
{
  float const scale = std::sqrt(Product(LastN<ND>(smap.dimensions())) / (float)Product(LastN<ND>(skern.dimensions())));
  smap.setZero();
  CxN<ND + 1> const     sk1 = skern.template chip<1>(c) * Cx(scale);
  CxNCMap<ND + 1> const sk1map(sk1.data(), sk1.dimensions());
  CxNMap<ND + 1>        smapmap(smap.data(), smap.dimensions());
  spad.forward(sk1map, smapmap);
  Sz<ND> ftd;
  std::iota(ftd.begin(), ftd.end(), 1);
  FFT::Adjoint(smap, ftd);
}

template <int ND> void NUFFTLowmem<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const     time = this->startForward(x, y, false);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), AddFront(LastN<ND>(workspace.dimensions()), workspace.dimension(0)));
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

template <int ND> void NUFFTLowmem<ND>::iforward(InCMap const &x, OutMap &y) const
{
  auto const     time = this->startForward(x, y, true);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), AddFront(LastN<ND>(workspace.dimensions()), workspace.dimension(0)));
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

template <int ND> void NUFFTLowmem<ND>::adjoint(OutCMap const &y, InMap &x) const
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

template <int ND> void NUFFTLowmem<ND>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const     time = this->startAdjoint(y, x, true);
  CxNMap<ND + 2> wsm(workspace.data(), workspace.dimensions());
  CxNMap<ND + 1> ws1m(workspace.data(), AddFront(LastN<ND>(workspace.dimensions()), workspace.dimension(0)));
  OutCMap        nc1m(nc1.data(), nc1.dimensions());
  for (Index ic = 0; ic < y.dimension(0); ic++) {
    kernToMap(ic);
    nc1.device(Threads::TensorDevice()) = y.slice(Sz3{ic, 0, 0}, Sz3{1, y.dimension(1), y.dimension(2)});
    gridder->adjoint(nc1m, wsm);
    FFT::Adjoint(workspace, fftDims);
    CxNMap<ND + 1> ws1m(workspace.data(), NoChannels(workspace.dimensions()));
    ws1m.device(Threads::TensorDevice()) = ws1m * smap.conjugate().broadcast(sbrd);
    x.device(Threads::TensorDevice()) += ws1m.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  }
  this->finishAdjoint(x, time, true);
}

template struct NUFFTLowmem<1>;
template struct NUFFTLowmem<2>;
template struct NUFFTLowmem<3>;

} // namespace rl::TOps
