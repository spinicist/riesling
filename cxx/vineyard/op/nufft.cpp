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

template <int NDim, bool VCC>
NUFFT<NDim, VCC>::NUFFT(GType::Ptr g, Sz<NDim> const matrix, Index const subgridSz)
  : Parent("NUFFT")
  , gridder{g}
  , workspace{gridder->ishape}
{
  if (std::equal(matrix.cbegin(), matrix.cend(), gridder->ishape.cbegin() + 2 + VCC, std::less_equal())) {
    ishape = Concatenate(FirstN<2 + VCC>(gridder->ishape), matrix);
  } else {
    throw Log::Failure("NUFFT", "Requested matrix {} but grid size is {}", matrix, LastN<NDim>(gridder->ishape));
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
  apo_ = Apodize(LastN<NDim>(ishape), LastN<NDim>(gridder->ishape), gridder->kernel).reshape(apo_shape); // Padding stuff
  fmt::print(stderr, "|apo_| {}\n", Norm(apo_));
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

template <int NDim, bool VCC>
auto NUFFT<NDim, VCC>::Make(TrajectoryN<NDim> const &traj,
                            std::string const       &ktype,
                            float const              osamp,
                            Index const              nChan,
                            Basis::CPtr              basis,
                            Sz<NDim> const           matrix,
                            Index const              subgridSz) -> std::shared_ptr<NUFFT<NDim, VCC>>
{
  auto g = TOps::Grid<NDim, VCC>::Make(traj, matrix, osamp, ktype, nChan, basis);
  return std::make_shared<NUFFT<NDim, VCC>>(g, matrix, subgridSz);
}

template <int NDim, bool VCC>
auto NUFFT<NDim, VCC>::Make(
  TrajectoryN<NDim> const &traj, GridOpts &opts, Index const nChan, Basis::CPtr basis, Sz<NDim> const matrix)
  -> std::shared_ptr<NUFFT<NDim, VCC>>
{
  auto g = TOps::Grid<NDim, VCC>::Make(traj, matrix, opts.osamp.Get(), opts.ktype.Get(), nChan, basis);
  return std::make_shared<NUFFT<NDim, VCC>>(g, matrix, opts.subgridSize.Get());
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->forward(workspace, y);
  this->finishForward(y, time, false);
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) = workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  fmt::print(stderr, "x {} ishape {} |x| {} |apo| {}\n", x.dimensions(), ishape, Norm(x), Norm(apo_));
  this->finishAdjoint(x, time, false);
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->iforward(workspace, y);
  this->finishForward(y, time, true);
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::iadjoint(OutCMap const &y, InMap &x) const
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

auto NUFFTAll(GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Index const       nTime,
              Basis::CPtr       basis,
              Sz3 const         shape) -> TOps::TOp<Cx, 6, 5>::Ptr
{
  if (gridOpts.lowmem) {
    throw Log::Failure("NUFFT", "Low memory option requires sensitivities");
  }
  if (gridOpts.vcc) {
    auto       nufft = TOps::NUFFT<3, true>::Make(traj, gridOpts, nC, basis, shape);
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
    auto nufft = TOps::NUFFT<3, false>::Make(traj, gridOpts, nC, basis, shape);
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
