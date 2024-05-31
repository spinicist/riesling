#include "nufft.hpp"

#include "../fft.hpp"
#include "kernel/kernel.hpp"
#include "loop.hpp"
#include "op/grid.hpp"

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
  , pad{AMin(matrix, LastN<NDim>(gridder.ishape)), gridder.ishape}
  , apo{pad.ishape, LastN<NDim>(gridder.ishape), gridder.kernel}
  , batches{nBatch}
{
  if (nChan % nBatch != 0) { Log::Fail("Batch size {} does not cleanly divide number of channels {}", nBatch, nChan); }
  ishape = pad.ishape;
  ishape[0] = nChan; // Undo batching
  oshape = gridder.oshape;
  oshape[0] = nChan;
  std::iota(fftDims.begin(), fftDims.end(), 2 + VCC);
  fftPh = FFT::PhaseShift(LastN<NDim>(gridder.ishape));
  Log::Debug("NUFFT Input Dims {} Output Dims {} Grid Dims {}", ishape, oshape, gridder.ishape);
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
    pad.forward(apo.forward(x), wsm);
    FFT::Forward(workspace, fftDims, fftPh);
    gridder.forward(workspace, y);
  } else {
    InTensor     xt(pad.ishape);
    OutTensor    yt(gridder.oshape);
    OutMap       ytm(yt.data(), yt.dimensions());
    Sz<NDim + 3> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder.ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      xt.device(Threads::GlobalDevice()) = x.slice(x_start, xt.dimensions());
      pad.forward(apo.forward(xt), wsm);
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
    apo.adjoint(pad.adjoint(workspace), x);
  } else {
    InTensor  xt(pad.ishape);
    InMap     xtm(xt.data(), xt.dimensions());
    OutTensor yt(gridder.oshape);

    Sz<NDim + 3> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder.ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      yt.device(Threads::GlobalDevice()) = y.slice(y_start, yt.dimensions());
      gridder.adjoint(yt, wsm);
      FFT::Adjoint(workspace, fftDims, fftPh);
      apo.adjoint(pad.adjoint(workspace), xtm);
      x.slice(x_start, xt.dimensions()).device(Threads::GlobalDevice()) = xt;
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
