#pragma once

#include "operator.hpp"

#include "apodize.hpp"
#include "fft.hpp"
#include "gridBase.hpp"
#include "pad.hpp"
#include "sdc.hpp"

namespace rl {
struct NUFFTOp final : Operator<5, 3>
{
  NUFFTOp(Sz3 const imgDims, GridBase<Cx> *g, SDCOp *sdc = nullptr)
    : gridder_{g}
    , fft_{gridder_->workspace(), g->mapping().type != Info::Type::TwoD}
    , pad_{Sz5{g->inputDimensions()[0], g->inputDimensions()[1], imgDims[0], imgDims[1], imgDims[2]}, g->inputDimensions()}
    , apo_{pad_.inputDimensions(), g}
    , sdc_{sdc}
  {
  }

  InputDims inputDimensions() const
  {
    return pad_.inputDimensions();
  }

  OutputDims outputDimensions() const
  {
    return gridder_->outputDimensions();
  }

  void calcToeplitz()
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

  template <typename T>
  auto A(T const &x) const
  {
    Log::Debug("Starting NUFFT forward");
    auto const &start = Log::Now();
    Output result(outputDimensions());
    result.device(Threads::GlobalDevice()) = gridder_->A(fft_.A(pad_.A(apo_.A(x))));
    Log::Debug("Finished NUFFT forward: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    Log::Debug("Starting NUFFT adjoint");
    auto const start = Log::Now();
    Input result(inputDimensions());
    if (sdc_) {
      result.device(Threads::GlobalDevice()) = apo_.Adj(pad_.Adj(fft_.Adj(gridder_->Adj(sdc_->Adj(x)))));
    } else {
      result.device(Threads::GlobalDevice()) = apo_.Adj(pad_.Adj(fft_.Adj(gridder_->Adj(x))));
    }
    Log::Debug("Finished NUFFT adjoint: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  Input AdjA(T const &x) const
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

  FFTOp<5> const &fft() const
  {
    return fft_;
  };

private:
  GridBase<Cx> *gridder_;
  FFTOp<5> fft_;
  PadOp<5> pad_;
  ApodizeOp<5> apo_;
  Cx5 tf_;
  SDCOp *sdc_;
};
} // namespace rieslin
