#include "nufft.hpp"

namespace rl {

NUFFTOp::NUFFTOp(Sz3 const imgDims, GridBase<Cx, 3> *g, SDCOp *sdc)
  : gridder_{g}
  , fft_{gridder_->workspace(), true}
  , pad_{Sz5{g->inputDimensions()[0], g->inputDimensions()[1], imgDims[0], imgDims[1], imgDims[2]}, g->inputDimensions()}
  , apo_{pad_.inputDimensions(), g}
  , sdc_{sdc}
{
  scale_ = 1.f;//std::sqrtf(float(Product(LastN<3>(gridder_->inputDimensions()))) / Product(LastN<3>(apo_.inputDimensions())));

  Log::Debug(
    "NUFFT Input Dims {} Output Dims {} Grid Dims {} Scale {}",
    inputDimensions(),
    outputDimensions(),
    gridder_->inputDimensions(),
    scale_);
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
  Log::Debug("NUFFT: Calculating Töplitz embedding");
  Sz5 const dims = AddFront(LastN<4>(gridder_->inputDimensions()), 1);
  tf_.resize(dims);
  tf_.setConstant(1.f);
  if (sdc_) {
    tf_ = gridder_->adjoint(sdc_->adjoint(gridder_->forward(tf_)));
  } else {
    tf_ = gridder_->adjoint(gridder_->forward(tf_));
  }
  Log::Tensor(Cx4(tf_.reshape(LastN<4>(dims))), "nufft-tf");
  Log::Debug(FMT_STRING("NUFFT: Calculated Töplitz. TF dimensions {}"), fmt::join(tf_.dimensions(), ","));
}

auto NUFFTOp::forward(Input const &x) const -> Output
{
  assert(x.dimensions() == inputDimensions());
  LOG_DEBUG("Starting NUFFT forward. Norm {}", Norm(x));
  auto const &start = Log::Now();
  Output result(outputDimensions());
  result.device(Threads::GlobalDevice()) = gridder_->forward(fft_.forward(pad_.forward(apo_.forward(x * x.constant(scale_)))));
  Log::Debug("Finished NUFFT forward: {}", Log::ToNow(start));
  LOG_DEBUG("Norm {}", Norm(result));
  return result;
}

auto NUFFTOp::adjoint(Output const &x) const -> Input
{
  assert(x.dimensions() == outputDimensions());
  LOG_DEBUG("Starting NUFFT adjoint. Norm {}", Norm(x));
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (sdc_) {
    result.device(Threads::GlobalDevice()) = apo_.adjoint(pad_.adjoint(fft_.adjoint(gridder_->adjoint(sdc_->adjoint(x)))));
  } else {
    result.device(Threads::GlobalDevice()) = apo_.adjoint(pad_.adjoint(fft_.adjoint(gridder_->adjoint(x))));
  }
  result.device(Threads::GlobalDevice()) = result * result.constant(scale_);
  Log::Debug("Finished NUFFT adjoint: {}", Log::ToNow(start));
  LOG_DEBUG("Norm {}", Norm(result));
  return result;
}

auto NUFFTOp::adjfwd(Input const &x) const -> Input
{
  Log::Debug("Starting NUFFT adjoint*forward");
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (tf_.size() == 0) {
    result.device(Threads::GlobalDevice()) = adjoint(forward(x));
  } else {
    Log::Debug("Using Töplitz embedding");
    Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brd;
    brd.set(0, this->inputDimensions()[0]);
    result.device(Threads::GlobalDevice()) = pad_.adjoint(fft_.adjoint(tf_.broadcast(brd) * fft_.forward(pad_.forward(x))));
  }
  Log::Debug("Finished NUFFT adjoint*forward: {}", Log::ToNow(start));
  return result;
}

auto NUFFTOp::fft() const -> FFTOp<5> const &
{
  return fft_;
};

} // namespace rl
