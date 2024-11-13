#include "nufft.hpp"

#include "../fft.hpp"
#include "apodize.hpp"
#include "log.hpp"
#include "op/top-impl.hpp"

#include "op/compose.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/reshape.hpp"

namespace rl::TOps {

template <int ND>
NUFFT<ND>::NUFFT(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  : Parent("NUFFT")
  , gridder{Grid<ND>::Make(opts, traj, nChan, basis)}
  , workspace{gridder->ishape}
{
  ishape = Concatenate(FirstN<2>(gridder->ishape), traj.matrixForFOV(opts.fov));
  oshape = gridder->oshape;
  std::iota(fftDims.begin(), fftDims.end(), 2);
  Log::Print("NUFFT", "ishape {} oshape {} grid {}", ishape, oshape, gridder->ishape);

  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  for (int ii = 0; ii < 2; ii++) {
    apo_shape[ii] = 1;
    apoBrd_[ii] = gridder->ishape[ii];
  }
  apo_ = Apodize(LastN<ND>(ishape), LastN<ND>(gridder->ishape), gridder->kernel).reshape(apo_shape); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 2; ii < InRank; ii++) {
    padLeft_[ii] = (gridder->ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder->ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND>
auto NUFFT<ND>::Make(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, Index const nChan, Basis::CPtr basis)
  -> std::shared_ptr<NUFFT<ND>>
{
  return std::make_shared<NUFFT<ND>>(opts, traj, nChan, basis);
}

template <int ND> void NUFFT<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->forward(workspace, y);
  this->finishForward(y, time, false);
}

template <int ND> void NUFFT<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) = workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, false);
}

template <int ND> void NUFFT<ND>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->iforward(workspace, y);
  this->finishForward(y, time, true);
}

template <int ND> void NUFFT<ND>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) += workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, true);
}

template struct NUFFT<1>;
template struct NUFFT<2>;
template struct NUFFT<3>;

auto NUFFTAll(Grid<3>::Opts const &gridOpts,
              Trajectory const    &traj,
              Index const          nC,
              Index const          nSlab,
              Index const          nTime,
              Basis::CPtr          basis) -> TOps::TOp<Cx, 6, 5>::Ptr
{
  auto nufft = TOps::NUFFT<3>::Make(gridOpts, traj, nC, basis);
  if (nSlab == 1) {
    auto reshape = TOps::MakeReshapeOutput(nufft, AddBack(nufft->oshape, 1));
    auto timeLoop = TOps::MakeLoop(reshape, nTime);
    return timeLoop;
  } else {
    auto loop = TOps::MakeLoop(nufft, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(nufft->ishape, nSlab);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop(compose1, nTime);
    return timeLoop;
  }
}

} // namespace rl::TOps
