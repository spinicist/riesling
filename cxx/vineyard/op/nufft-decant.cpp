#include "nufft-decant.hpp"

#include "../fft.hpp"
#include "apodize.hpp"
#include "log.hpp"
#include "op/top-impl.hpp"

namespace rl::TOps {

template <int ND>
NUFFTDecant<ND>::NUFFTDecant(GType::Ptr g, Sz<ND> const matrix)
  : Parent("NUFFTDecant")
  , gridder{g}
  , workspace{gridder->ishape}
{
  if (std::equal(matrix.cbegin(), matrix.cend(), gridder->ishape.cbegin() + 1, std::less_equal())) {
    ishape = Concatenate(FirstN<1>(gridder->ishape), matrix);
  } else {
    throw Log::Failure("NUFFTDecant", "Requested matrix {} but grid size is {}", matrix, LastN<ND>(gridder->ishape));
  }
  oshape = gridder->oshape;
  std::iota(fftDims.begin(), fftDims.end(), 1);
  Log::Print("NUFFTDecant", "ishape {} oshape {} grid {}", ishape, oshape, gridder->ishape);

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
    padLeft_[ii] = (gridder->ishape[ii] - ishape[ii] + 1) / 2;
    padRight[ii] = (gridder->ishape[ii] - ishape[ii]) / 2;
  }
  std::transform(padLeft_.cbegin(), padLeft_.cend(), padRight.cbegin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int ND>
auto NUFFTDecant<ND>::Make(TrajectoryN<ND> const &traj,
                           std::string const     &ktype,
                           float const            osamp,
                           CxN<ND + 2> const     &skern,
                           Basis::CPtr            basis,
                           Sz<ND> const           matrix,
                           Index const            subgridSz) -> std::shared_ptr<NUFFTDecant<ND>>
{
  auto g = GType::Make(traj, matrix, osamp, ktype, skern, basis);
  return std::make_shared<NUFFTDecant<ND>>(g, matrix);
}

template <int ND>
auto NUFFTDecant<ND>::Make(
  TrajectoryN<ND> const &traj, GridOpts<ND> const &opts, CxN<ND + 2> const &skern, Basis::CPtr basis, Sz<ND> const matrix)
  -> std::shared_ptr<NUFFTDecant<ND>>
{
  auto g = GType::Make(traj, matrix, opts.osamp, opts.ktype, skern, basis);
  return std::make_shared<NUFFTDecant<ND>>(g, matrix);
}

template <int ND> void NUFFTDecant<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->forward(workspace, y);
  this->finishForward(y, time, false);
}

template <int ND> void NUFFTDecant<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) = workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, false);
}

template <int ND> void NUFFTDecant<ND>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  wsm.device(Threads::TensorDevice()) = (x * apo_.broadcast(apoBrd_)).pad(paddings_);
  FFT::Forward(workspace, fftDims);
  gridder->iforward(workspace, y);
  this->finishForward(y, time, true);
}

template <int ND> void NUFFTDecant<ND>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  InMap      wsm(workspace.data(), gridder->ishape);
  gridder->adjoint(y, wsm);
  FFT::Adjoint(workspace, fftDims);
  x.device(Threads::TensorDevice()) += workspace.slice(padLeft_, ishape) * apo_.broadcast(apoBrd_);
  this->finishAdjoint(x, time, true);
}

template struct NUFFTDecant<1>;
template struct NUFFTDecant<2>;
template struct NUFFTDecant<3>;

} // namespace rl::TOps
