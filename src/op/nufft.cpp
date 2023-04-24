#include "nufft.hpp"

#include "loop.hpp"
#include "rank.hpp"

namespace rl {

template <size_t NDim>
NUFFTOp<NDim>::NUFFTOp(
  std::shared_ptr<GridBase<Cx, NDim>> g, Sz<NDim> const matrix, std::shared_ptr<TensorOperator<Cx, 3>> s, bool toeplitz)
  : Parent("NUFFTOp", Concatenate(FirstN<2>(g->ishape), AMin(matrix, LastN<NDim>(g->ishape))), g->oshape)
  , gridder{g}
  , workspace{gridder->ishape}
  , fft{FFT::Make<NDim + 2, NDim>(workspace)}
  , pad{AMin(matrix, LastN<NDim>(gridder->ishape)), gridder->ishape}
  , apo{pad.ishape, gridder.get()}
  , sdc{s ? s : std::make_shared<TensorIdentity<Cx, 3>>(gridder->oshape)}
{
  Log::Print<Log::Level::High>("NUFFT Input Dims {} Output Dims {} Grid Dims {}", ishape, oshape, gridder->ishape);
  if (toeplitz) {
    Log::Print("Calculating Töplitz embedding");
    tf_.resize(ishape);
    tf_.setConstant(1.f);
    tf_ = adjoint(sdc->forward(forward(tf_)));
  }
}

template <size_t NDim>
void NUFFTOp<NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  InMap wsm(workspace.data(), gridder->ishape);
  pad.forward(apo.forward(x), wsm);
  fft->forward(workspace);
  gridder->forward(workspace, y);
  this->finishForward(y, time);
}

template <size_t NDim>
void NUFFTOp<NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  InMap wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(sdc->adjoint(y), wsm);
  fft->reverse(workspace);
  apo.adjoint(pad.adjoint(workspace), x);
  this->finishAdjoint(x, time);
}

// template <size_t NDim>
// auto NUFFTOp<NDim>::adjfwd(InputMap x) const -> InputMap
// {
//   if (tf_.size() == 0) {
//     return adjoint(forward(x));
//   } else {
//     auto temp = fft_.forward(pad_.forward(x));
//     temp *= tf_;
//     return pad_.adjoint(fft_.adjoint(temp));
//   }
// }

template struct NUFFTOp<1>;
template struct NUFFTOp<2>;
template struct NUFFTOp<3>;

std::shared_ptr<TensorOperator<Cx, 5, 4>> make_nufft(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Sz3 const matrix,
  std::optional<Re2> basis,
  std::shared_ptr<TensorOperator<Cx, 3>> sdc,
  bool const toeplitz)
{
  if (traj.nDims() == 2) {
    Log::Print<Log::Level::Debug>("Creating 2D Multi-slice NUFFT");
    auto grid = make_grid<Cx, 2>(traj, ktype, osamp * (toeplitz ? 2.f : 1.f), nC, basis);
    auto nufft2 = std::make_shared<NUFFTOp<2>>(grid, FirstN<2>(matrix), sdc, toeplitz);
    return std::make_shared<LoopOp<NUFFTOp<2>>>(nufft2, traj.info().matrix[2]);
  } else {
    Log::Print<Log::Level::Debug>("Creating full 3D NUFFT");
    auto grid = make_grid<Cx, 3>(traj, ktype, osamp * (toeplitz ? 2.f : 1.f), nC, basis);
    return std::make_shared<IncreaseOutputRank<NUFFTOp<3>>>(std::make_shared<NUFFTOp<3>>(grid, matrix, sdc, toeplitz));
  }
}

} // namespace rl
