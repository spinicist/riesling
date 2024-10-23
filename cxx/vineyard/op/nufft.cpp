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

template <int ND, bool VCC>
NUFFT<ND, VCC>::NUFFT(GType::Ptr g, Sz<ND> const matrix, Index const subgridSz)
  : Parent("NUFFT")
  , gridder{g}
  , workspace{gridder->ishape}
{
  if (std::equal(matrix.cbegin(), matrix.cend(), gridder->ishape.cbegin() + 2 + VCC, std::less_equal())) {
    ishape = Concatenate(FirstN<2 + VCC>(gridder->ishape), matrix);
  } else {
    throw Log::Failure("NUFFT", "Requested matrix {} but grid size is {}", matrix, LastN<ND>(gridder->ishape));
  }
  oshape = gridder->oshape;
  std::iota(fftDims.begin(), fftDims.end(), 2 + VCC);
  Log::Print("NUFFT", "ishape {} oshape {} grid {}", ishape, oshape, gridder->ishape);

  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  for (int ii = 0; ii < 2 + VCC; ii++) {
    apo_shape[ii] = 1;
    apoBrd_[ii] = gridder->ishape[ii];
  }
  apo_ = Apodize(LastN<ND>(ishape), LastN<ND>(gridder->ishape), gridder->kernel).reshape(apo_shape); // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 2 + VCC; ii < InRank; ii++) {
    padLeft_[ii] = (gridder->ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder->ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND, bool VCC>
auto NUFFT<ND, VCC>::Make(TrajectoryN<ND> const &traj,
                          std::string const     &ktype,
                          float const            osamp,
                          Index const            nChan,
                          Basis::CPtr            basis,
                          Sz<ND> const           matrix,
                          Index const            subgridSz) -> std::shared_ptr<NUFFT<ND, VCC>>
{
  auto g = TOps::Grid<ND, VCC>::Make(traj, matrix, osamp, ktype, nChan, basis);
  return std::make_shared<NUFFT<ND, VCC>>(g, matrix, subgridSz);
}

template <int ND, bool VCC>
auto NUFFT<ND, VCC>::Make(TrajectoryN<ND> const &traj, GridOpts<ND> const &opts, Index const nChan, Basis::CPtr basis)
  -> std::shared_ptr<NUFFT<ND, VCC>>
{
  auto const matrix = traj.matrixForFOV(opts.fov);
  auto       g = TOps::Grid<ND, VCC>::Make(traj, matrix, opts.osamp, opts.ktype, nChan, basis);
  return std::make_shared<NUFFT<ND, VCC>>(g, matrix, opts.subgridSize);
}

template <int ND, bool VCC> void NUFFT<ND, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->forward(workspace, y);
  this->finishForward(y, time, false);
}

template <int ND, bool VCC> void NUFFT<ND, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) = workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, false);
}

template <int ND, bool VCC> void NUFFT<ND, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->iforward(workspace, y);
  this->finishForward(y, time, true);
}

template <int ND, bool VCC> void NUFFT<ND, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) += workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, true);
}

template struct NUFFT<1, false>;
template struct NUFFT<2, false>;
template struct NUFFT<3, false>;

template struct NUFFT<1, true>;
template struct NUFFT<2, true>;
template struct NUFFT<3, true>;

auto NUFFTAll(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nC, Index const nSlab, Index const nTime, Basis::CPtr basis)
  -> TOps::TOp<Cx, 6, 5>::Ptr
{
  if (gridOpts.vcc) {
    auto       nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, nC, basis);
    auto const ns = nufft->ishape;
    auto       reshape = TOps::MakeReshapeInput(nufft, Sz5{ns[0] * ns[1], ns[2], ns[3], ns[4], ns[5]});
    if (nSlab == 1) {
      auto rout = TOps::MakeReshapeOutput(reshape, AddBack(reshape->oshape, 1));
      auto timeLoop = TOps::MakeLoop(rout, nTime);
      return timeLoop;
    } else {
      auto loop = TOps::MakeLoop(reshape, nSlab);
      auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(reshape->ishape, nSlab);
      auto compose2 = TOps::MakeCompose(slabToVol, loop);
      auto timeLoop = TOps::MakeLoop(compose2, nTime);
      return timeLoop;
    }
  } else {
    auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, nC, basis);
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
}

} // namespace rl::TOps
