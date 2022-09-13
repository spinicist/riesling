#include "nufft.hpp"

namespace rl {

NUFFTOp::NUFFTOp(Sz3 const imgDims, GridBase<Cx, 3> *g, SDCOp *sdc)
  : gridder_{g}
  , fft_{gridder_->workspace(), true}
  , pad_{AddFront(imgDims, g->inputDimensions()[0], g->inputDimensions()[1]), LastN<3>(g->inputDimensions())}
  , apo_{pad_.inputDimensions(), g}
  , sdc_{sdc}
{
  Log::Print<Log::Level::High>(
    "NUFFT Input Dims {} Output Dims {} Grid Dims {}",
    inputDimensions(),
    outputDimensions(),
    gridder_->inputDimensions());
}

auto NUFFTOp::inputDimensions() const -> InputDims
{
  return apo_.inputDimensions();
}

auto NUFFTOp::outputDimensions() const -> OutputDims
{
  return gridder_->outputDimensions();
}

void NUFFTOp::calcToeplitz()
{
  Log::Print("NUFFT: Calculating Töplitz embedding");
  Sz5 const dims = AddFront(LastN<4>(gridder_->inputDimensions()), 1);
  tf_.resize(dims);
  tf_.setConstant(1.f);
  if (sdc_) {
    tf_ = gridder_->adjoint(sdc_->adjoint(gridder_->forward(tf_)));
  } else {
    tf_ = gridder_->adjoint(gridder_->forward(tf_));
  }
}

auto NUFFTOp::forward(Input const &x) const -> Output
{
  assert(x.dimensions() == inputDimensions());
  auto const &start = Log::Now();
  Output result(outputDimensions());
  result.device(Threads::GlobalDevice()) =
    gridder_->forward(fft_.forward(pad_.forward(apo_.forward(x))));
  LOG_DEBUG("NUFFT Forward. Norm {}->{}. Time {}", Norm(x), Norm(result), Log::ToNow(start));
  return result;
}

auto NUFFTOp::adjoint(Output const &x) const -> Input
{
  assert(x.dimensions() == outputDimensions());
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (sdc_) {
    result.device(Threads::GlobalDevice()) =
      apo_.adjoint(pad_.adjoint(fft_.adjoint(gridder_->adjoint(sdc_->adjoint(x)))));
  } else {
    result.device(Threads::GlobalDevice()) = apo_.adjoint(pad_.adjoint(fft_.adjoint(gridder_->adjoint(x))));
  }
  result.device(Threads::GlobalDevice()) = result;
  LOG_DEBUG("NUFFT Adjoint. Norm {}->{}. Time {}", Norm(x), Norm(result), Log::ToNow(start));
  return result;
}

auto NUFFTOp::adjfwd(Input const &x) const -> Input
{
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (tf_.size() == 0) {
    result.device(Threads::GlobalDevice()) = adjoint(forward(x));
  } else {
    Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brd;
    brd.set(0, this->inputDimensions()[0]);
    result.device(Threads::GlobalDevice()) =
      pad_.adjoint(fft_.adjoint(tf_.broadcast(brd) * fft_.forward(pad_.forward(x))));
  }
  LOG_DEBUG("Finished NUFFT adjoint*forward. Norm {}->{}. Time {}", Norm(x), Norm(result), Log::ToNow(start));
  return result;
}

auto NUFFTOp::fft() const -> FFTOp<5> const &
{
  return fft_;
};

} // namespace rl
