#include "nufft.hpp"

namespace rl {

NUFFTOp::NUFFTOp(Sz3 const imgDims, GridBase<Cx> *g, SDCOp *sdc)
  : gridder_{g}
  , fft_{gridder_->workspace(), g->mapping().type != Info::Type::TwoD}
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
    tf_ = gridder_->Adj(sdc_->Adj(gridder_->A(tf_)));
  } else {
    tf_ = gridder_->Adj(gridder_->A(tf_));
  }
  Log::Tensor(Cx4(tf_.reshape(LastN<4>(dims))), "nufft-tf");
  Log::Debug(FMT_STRING("NUFFT: Calculated Töplitz. TF dimensions {}"), fmt::join(tf_.dimensions(), ","));
}

auto NUFFTOp::A(Input const &x) const -> Output
{
  assert(x.dimensions() == inputDimensions());
  LOG_DEBUG("Starting NUFFT forward. Norm {}", Norm(x));
  auto const &start = Log::Now();
  Output result(outputDimensions());
  result.device(Threads::GlobalDevice()) = gridder_->A(fft_.A(pad_.A(apo_.A(x * x.constant(scale_)))));
  Log::Debug("Finished NUFFT forward: {}", Log::ToNow(start));
  LOG_DEBUG("Norm {}", Norm(result));
  return result;
}

auto NUFFTOp::Adj(Output const &x) const -> Input
{
  assert(x.dimensions() == outputDimensions());
  LOG_DEBUG("Starting NUFFT adjoint. Norm {}", Norm(x));
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (sdc_) {
    result.device(Threads::GlobalDevice()) = apo_.Adj(pad_.Adj(fft_.Adj(gridder_->Adj(sdc_->Adj(x)))));
  } else {
    result.device(Threads::GlobalDevice()) = apo_.Adj(pad_.Adj(fft_.Adj(gridder_->Adj(x))));
  }
  result.device(Threads::GlobalDevice()) = result * result.constant(scale_);
  Log::Debug("Finished NUFFT adjoint: {}", Log::ToNow(start));
  LOG_DEBUG("Norm {}", Norm(result));
  return result;
}

auto NUFFTOp::AdjA(Input const &x) const -> Input
{
  Log::Debug("Starting NUFFT adjoint*forward");
  auto const start = Log::Now();
  Input result(inputDimensions());
  if (tf_.size() == 0) {
    result.device(Threads::GlobalDevice()) = Adj(A(x));
  } else {
    Log::Debug("Using Töplitz embedding");
    Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> brd;
    brd.set(0, this->inputDimensions()[0]);
    result.device(Threads::GlobalDevice()) = pad_.Adj(fft_.Adj(tf_.broadcast(brd) * fft_.A(pad_.A(x))));
  }
  Log::Debug("Finished NUFFT adjoint*forward: {}", Log::ToNow(start));
  return result;
}

auto NUFFTOp::fft() const -> FFTOp<5> const &
{
  return fft_;
};

} // namespace rl
