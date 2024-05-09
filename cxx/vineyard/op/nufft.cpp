#include "nufft.hpp"

#include "kernel/kernel.hpp"
#include "loop.hpp"
#include "op/grid.hpp"
#include "rank.hpp"

namespace rl {

template <int NDim>
NUFFTOp<NDim>::NUFFTOp(std::shared_ptr<Grid<Cx, NDim>>        g,
                       Sz<NDim> const                         matrix,
                       Index const                            nB)
  : Parent("NUFFTOp",
           AddFront(AMin(matrix, LastN<NDim>(g->ishape)), g->ishape[0] * nB, g->ishape[1]),
           AddFront(LastN<2>(g->oshape), g->oshape[0] * nB))
  , gridder{g}
  , workspace{gridder->ishape}
  , fft{FFT::Make<NDim + 2, NDim>(workspace)}
  , pad{AMin(matrix, LastN<NDim>(gridder->ishape)), gridder->ishape}
  , apo{pad.ishape, LastN<NDim>(gridder->ishape), gridder->kernel}
  , batches{nB}
{
  Log::Debug("NUFFT Input Dims {} Output Dims {} Grid Dims {}", ishape, oshape, gridder->ishape);
}

template <int NDim>
void NUFFTOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  InMap      wsm(workspace.data(), gridder->ishape);
  if (batches == 1) {
    pad.forward(apo.forward(x), wsm);
    fft->forward(workspace);
    gridder->forward(workspace, y);
  } else {
    InTensor     xt(pad.ishape);
    OutTensor    yt(gridder->oshape);
    OutMap       ytm(yt.data(), yt.dimensions());
    Sz<NDim + 2> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder->ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      xt.device(Threads::GlobalDevice()) = x.slice(x_start, xt.dimensions());
      pad.forward(apo.forward(xt), wsm);
      fft->forward(workspace);
      gridder->forward(workspace, ytm);
      y.slice(y_start, yt.dimensions()).device(Threads::GlobalDevice()) = yt;
    }
  }
  this->finishForward(y, time);
}

template <int NDim>
void NUFFTOp<NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  InMap      wsm(workspace.data(), gridder->ishape);
  if (batches == 1) {
    gridder->adjoint(y, wsm);
    fft->reverse(workspace);
    apo.adjoint(pad.adjoint(workspace), x);
  } else {
    InTensor  xt(pad.ishape);
    InMap     xtm(xt.data(), xt.dimensions());
    OutTensor yt(gridder->oshape);

    Sz<NDim + 2> x_start;
    Sz3          y_start;
    for (Index ib = 0; ib < batches; ib++) {
      Index const ic = ib * gridder->ishape[0];
      x_start[0] = ic;
      y_start[0] = ic;
      yt.device(Threads::GlobalDevice()) = y.slice(y_start, yt.dimensions());
      gridder->adjoint(yt, wsm);
      fft->reverse(workspace);
      apo.adjoint(pad.adjoint(workspace), xtm);
      x.slice(x_start, xt.dimensions()).device(Threads::GlobalDevice()) = xt;
    }
  }
  this->finishAdjoint(x, time);
}

template struct NUFFTOp<1>;
template struct NUFFTOp<2>;
template struct NUFFTOp<3>;

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_nufft(Trajectory const                      &traj,
                                                     GridOpts                              &opts,
                                                     Index const                            nC,
                                                     Sz3 const                              matrix,
                                                     Basis<Cx> const                       &basis)
{
  Index batchSize = nC;
  if (opts.batches) {
    if (nC % opts.batches.Get() != 0) {
      Log::Fail("Batch size {} does not cleanly divide number of channels {}", opts.batches.Get(), nC);
    }
    batchSize /= opts.batches.Get();
    Log::Print("Using {} batches size {}", opts.batches.Get(), batchSize);
  }
  Log::Debug("Creating NUFFT");
  auto grid =
    Grid<Cx, 3>::Make(traj, opts.ktype.Get(), opts.osamp.Get(), batchSize, basis, opts.bucketSize.Get(), opts.splitSize.Get());
  return std::make_shared<IncreaseOutputRank<NUFFTOp<3>>>(std::make_shared<NUFFTOp<3>>(grid, matrix, opts.batches.Get()));
}

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_nufft(Trajectory const                      &traj,
                                                     std::string const                     &ktype,
                                                     float const                            osamp,
                                                     Index const                            nC,
                                                     Sz3 const                              matrix,
                                                     Basis<Cx> const                       &basis,
                                                     Index const                            bSz,
                                                     Index const                            sSz)
{
  Log::Debug("Creating 3D NUFFT");
  auto grid = Grid<Cx, 3>::Make(traj, ktype, osamp, nC, basis, bSz, sSz);
  return std::make_shared<IncreaseOutputRank<NUFFTOp<3>>>(std::make_shared<NUFFTOp<3>>(grid, matrix, 1));
}

} // namespace rl
