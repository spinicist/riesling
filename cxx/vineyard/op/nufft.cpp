#include "nufft.hpp"

#include "../fft.hpp"
#include "apodize.hpp"

namespace rl::TOps {

template <int NDim, bool VCC>
NUFFT<NDim, VCC>::NUFFT(Sz<NDim> const           matrix,
                        TrajectoryN<NDim> const &traj,
                        std::string const       &ktype,
                        float const              osamp,
                        Index const              nChan,
                        Basis<Cx> const         &basis,
                        Index const              subgridSz,
                        Index const              splitSz,
                        Index const              nBatch)
  : Parent("NUFFT")
  , gridder{traj, ktype, osamp, nChan / nBatch, basis, subgridSz, splitSz}
  , workspace{gridder.ishape}
  , batches{nBatch}
{
  if (nChan % nBatch != 0) { Log::Fail("Batch size {} does not cleanly divide number of channels {}", nBatch, nChan); }
  // We need to take the minimum of the image dimensions, then stitch the right dimensions onto the front
  batchShape_ = Concatenate(FirstN<2 + VCC>(gridder.ishape), AMin(matrix, LastN<NDim>(gridder.ishape)));
  ishape = batchShape_;
  ishape[0] = nChan; // Undo batching
  oshape = gridder.oshape;
  oshape[0] = nChan;
  std::iota(fftDims.begin(), fftDims.end(), 2 + VCC);
  fftPh = FFT::PhaseShift(LastN<NDim>(gridder.ishape));
  Log::Print("NUFFT Input {} Output {} Grid {} Batches {}", ishape, oshape, gridder.ishape, batches);

  // Calculate apodization correction
  auto apo_shape = ishape;
  apoBrd_.fill(1);
  for (int ii = 0; ii < 2 + VCC; ii++) {
    apo_shape[ii] = 1;
    apoBrd_[ii] = gridder.ishape[ii];
  }
  apo_ = Apodize(LastN<NDim>(ishape), LastN<NDim>(gridder.ishape), gridder.kernel).reshape(apo_shape);

  // Padding stuff
  Sz<InRank> padRight;
  padLeft_.fill(0);
  padRight.fill(0);
  for (int ii = 2 + VCC; ii < InRank; ii++) {
    padLeft_[ii] = (gridder.ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder.ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int NDim, bool VCC>
auto NUFFT<NDim, VCC>::Make(Sz<NDim> const           matrix,
                            TrajectoryN<NDim> const &traj,
                            GridOpts                &opts,
                            Index const              nC,
                            Basis<Cx> const         &basis) -> std::shared_ptr<NUFFT<NDim, VCC>>
{
  return std::make_shared<NUFFT<NDim, VCC>>(matrix, traj, opts.ktype.Get(), opts.osamp.Get(), nC, basis, opts.subgridSize.Get(),
                                            opts.splitSize.Get(), opts.batches.Get());
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  InMap      wsm(workspace.data(), gridder.ishape);
  if (batches == 1) {
    wsm.device(Threads::GlobalDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
    FFT::Forward(workspace, fftDims, fftPh);
    gridder.forward(workspace, y);
  } else {
    OutTensor    yt(gridder.oshape);
    OutMap       ytm(yt.data(), yt.dimensions());
    Sz<NDim + 3> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder.ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      wsm.device(Threads::GlobalDevice()) = (x.slice(x_start, batchShape_) * apo_.broadcast(apoBrd_)).pad(paddings_);
      FFT::Forward(workspace, fftDims, fftPh);
      gridder.forward(workspace, ytm);
      y.slice(y_start, yt.dimensions()).device(Threads::GlobalDevice()) = yt;
    }
  }
  this->finishForward(y, time);
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  InMap      wsm(workspace.data(), gridder.ishape);
  if (batches == 1) {
    gridder.adjoint(y, wsm);
    FFT::Adjoint(workspace, fftDims, fftPh);
    x.device(Threads::GlobalDevice()) = workspace.slice(padLeft_, batchShape_) * apo_.broadcast(apoBrd_);
  } else {
    OutTensor    yt(gridder.oshape);
    Sz<NDim + 3> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder.ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      yt.device(Threads::GlobalDevice()) = y.slice(y_start, yt.dimensions());
      gridder.adjoint(yt, wsm);
      FFT::Adjoint(workspace, fftDims, fftPh);
      x.slice(x_start, batchShape_).device(Threads::GlobalDevice()) =
        workspace.slice(padLeft_, batchShape_) * apo_.broadcast(apoBrd_);
    }
  }
  this->finishAdjoint(x, time);
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y);
  InMap      wsm(workspace.data(), gridder.ishape);
  if (batches == 1) {
    wsm.device(Threads::GlobalDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
    FFT::Forward(workspace, fftDims, fftPh);
    gridder.iforward(workspace, y);
  } else {
    OutTensor    yt(gridder.oshape);
    OutMap       ytm(yt.data(), yt.dimensions());
    Sz<NDim + 3> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder.ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      wsm.device(Threads::GlobalDevice()) = (x.slice(x_start, batchShape_) * apo_.broadcast(apoBrd_)).pad(paddings_);
      FFT::Forward(workspace, fftDims, fftPh);
      gridder.forward(workspace, ytm);
      y.slice(y_start, yt.dimensions()).device(Threads::GlobalDevice()) += yt;
    }
  }
  this->finishForward(y, time);
}

template <int NDim, bool VCC> void NUFFT<NDim, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x);
  InMap      wsm(workspace.data(), gridder.ishape);
  if (batches == 1) {
    gridder.adjoint(y, wsm);
    FFT::Adjoint(workspace, fftDims, fftPh);
    x.device(Threads::GlobalDevice()) += workspace.slice(padLeft_, batchShape_) * apo_.broadcast(apoBrd_);
  } else {
    OutTensor    yt(gridder.oshape);
    Sz<NDim + 3> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder.ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      yt.device(Threads::GlobalDevice()) = y.slice(y_start, yt.dimensions());
      gridder.adjoint(yt, wsm);
      FFT::Adjoint(workspace, fftDims, fftPh);
      x.slice(x_start, batchShape_).device(Threads::GlobalDevice()) +=
        workspace.slice(padLeft_, batchShape_) * apo_.broadcast(apoBrd_);
    }
  }
  this->finishAdjoint(x, time);
}


template struct NUFFT<1, false>;
template struct NUFFT<2, false>;
template struct NUFFT<3, false>;

template struct NUFFT<1, true>;
template struct NUFFT<2, true>;
template struct NUFFT<3, true>;

} // namespace rl::TOps
