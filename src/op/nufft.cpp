#include "nufft.hpp"

#include "loop.hpp"
#include "rank.hpp"

namespace rl {

template <size_t NDim>
NUFFTOp<NDim>::NUFFTOp(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Sz<NDim> const matrix,
  Operator<3, 3> *sdc,
  std::optional<Re2> basis,
  bool toeplitz)
  : gridder_{make_grid<Cx, NDim>(traj, ktype, osamp * (toeplitz ? 2.f: 1.f), nC, basis)}
  , fft_{gridder_->workspace()}
  , pad_{matrix, LastN<NDim>(gridder_->inputDimensions()), FirstN<2>(gridder_->inputDimensions())}
  , apo_{pad_.inputDimensions(), gridder_.get()}
  , sdc_{sdc}
{
  Log::Print<Log::Level::High>(
    "NUFFT Input Dims {} Output Dims {} Grid Dims {}", inputDimensions(), outputDimensions(), gridder_->inputDimensions());
  if (toeplitz) {
    Log::Print("Calculating Töplitz embedding");
    tf_.resize(inputDimensions());
    tf_.setConstant(1.f);
    if (sdc_) {
      tf_ = adjoint(sdc_->adjoint(forward(tf_)));
    } else {
      tf_ = adjoint(forward(tf_));
    }
  }
}

template <size_t NDim>
auto NUFFTOp<NDim>::inputDimensions() const -> InputDims
{
  return apo_.inputDimensions();
}

template <size_t NDim>
auto NUFFTOp<NDim>::outputDimensions() const -> OutputDims
{
  return gridder_->outputDimensions();
}

template <size_t NDim>
auto NUFFTOp<NDim>::forward(Input const &x) const -> Output const &
{
  assert(x.dimensions() == inputDimensions());
  return gridder_->forward(fft_.forward(pad_.forward(apo_.forward(x))));
}

template <size_t NDim>
auto NUFFTOp<NDim>::adjoint(Output const &y) const -> Input const &
{
  assert(y.dimensions() == outputDimensions());
  if (sdc_) {
    return apo_.adjoint(pad_.adjoint(fft_.adjoint(gridder_->adjoint(sdc_->adjoint(y)))));
  } else {
    return apo_.adjoint(pad_.adjoint(fft_.adjoint(gridder_->adjoint(y))));
  }
}

template <size_t NDim>
auto NUFFTOp<NDim>::adjfwd(Input const &x) const -> Input
{
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (tf_.size() == 0) {
    result.device(Threads::GlobalDevice()) = adjoint(forward(x));
  } else {
    result.device(Threads::GlobalDevice()) = fft_.adjoint(tf_ * fft_.forward(pad_.forward(x)));
  }
  LOG_DEBUG("Finished NUFFT adjoint*forward. Norm {}->{}. Time {}", Norm(x), Norm(result), Log::ToNow(start));
  return result;
}

template <size_t NDim>
auto NUFFTOp<NDim>::fft() const -> FFTOp<NDim + 2, NDim> const &
{
  return fft_;
};

template struct NUFFTOp<2>;
template struct NUFFTOp<3>;

std::unique_ptr<Operator<5, 4>> make_nufft(
  Trajectory const &traj,
  std::string const &ktype,
  float const osamp,
  Index const nC,
  Sz3 const matrix,
  Operator<3, 3> *sdc,
  std::optional<Re2> basis,
  bool const toeplitz)
{
  if (traj.nDims() == 2) {
    Log::Print<Log::Level::Debug>("Creating 2D Multi-slice NUFFT");
    NUFFTOp<2> nufft2(traj, ktype, osamp, nC, FirstN<2>(matrix), sdc, basis, toeplitz);
    return std::make_unique<LoopOp<NUFFTOp<2>>>(nufft2, traj.info().matrix[2]);
  } else {
    Log::Print<Log::Level::Debug>("Creating full 3D NUFFT");
    NUFFTOp<3> nufft3(traj, ktype, osamp, nC, matrix, sdc, basis, toeplitz);
    return std::make_unique<IncreaseOutputRank<NUFFTOp<3>>>(nufft3);
  }
}

} // namespace rl
