#include "nufft.hpp"

#include "kernel/kernel.hpp"
#include "loop.hpp"
#include "op/grid.hpp"
#include "rank.hpp"

namespace rl {

template <int NDim>
NUFFTOp<NDim>::NUFFTOp(std::shared_ptr<Grid<Cx, NDim>> g, Sz<NDim> const matrix, std::shared_ptr<TensorOperator<Cx, 3>> s)
  : Parent("NUFFTOp", Concatenate(FirstN<2>(g->ishape), AMin(matrix, LastN<NDim>(g->ishape))), g->oshape)
  , gridder{g}
  , workspace{gridder->ishape}
  , fft{FFT::Make<NDim + 2, NDim>(workspace)}
  , pad{AMin(matrix, LastN<NDim>(gridder->ishape)), gridder->ishape}
  , apo{pad.ishape, LastN<NDim>(gridder->ishape), gridder->kernel}
  , sdc{s ? s : std::make_shared<TensorIdentity<Cx, 3>>(gridder->oshape)}
{
  Log::Debug("NUFFT Input Dims {} Output Dims {} Grid Dims {}", ishape, oshape, gridder->ishape);
}

template <int NDim>
void NUFFTOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  InMap      wsm(workspace.data(), gridder->ishape);
  pad.forward(apo.forward(x), wsm);
  // Log::Tensor("nufft-fwd", pad.oshape, wsm.data());
  fft->forward(workspace);
  gridder->forward(workspace, y);
  this->finishForward(y, time);
}

template <int NDim>
void NUFFTOp<NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(sdc->adjoint(y), wsm);
  fft->reverse(workspace);
  // Log::Tensor("nufft-adj", pad.oshape, wsm.data());
  apo.adjoint(pad.adjoint(workspace), x);
  this->finishAdjoint(x, time);
}

template struct NUFFTOp<1>;
template struct NUFFTOp<2>;
template struct NUFFTOp<3>;

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_nufft(Trajectory const                      &traj,
                                                     std::string const                     &ktype,
                                                     float const                            osamp,
                                                     Index const                            nC,
                                                     Sz3 const                              matrix,
                                                     Basis<Cx> const                       &basis,
                                                     std::shared_ptr<TensorOperator<Cx, 3>> sdc,
                                                     Index const                            bSz,
                                                     Index const                            sSz)
{
  if (traj.nDims() == 2) {
    Log::Debug("Creating 2D Multi-slice NUFFT");
    auto grid = Grid<Cx, 2>::Make(traj, ktype, osamp, nC, basis, bSz, sSz);
    auto nufft2 = std::make_shared<NUFFTOp<2>>(grid, FirstN<2>(matrix), sdc);
    return std::make_shared<LoopOp<NUFFTOp<2>>>(nufft2, traj.info().matrix[2]);
  } else {
    Log::Debug("Creating full 3D NUFFT");
    auto grid = Grid<Cx, 3>::Make(traj, ktype, osamp, nC, basis, bSz, sSz);
    return std::make_shared<IncreaseOutputRank<NUFFTOp<3>>>(std::make_shared<NUFFTOp<3>>(grid, matrix, sdc));
  }
}

} // namespace rl
