#pragma once

#include "operator.h"

#include "apodize.hpp"
#include "fft.hpp"
#include "grid.h"
#include "pad.hpp"

struct NUFFTOp final : Operator<5, 3>
{
  NUFFTOp(Sz3 const imgDims, GridBase *g)
    : gridder_{g}
    , fft_{gridder_->workspace()}
    , pad_{Sz5{g->inputDimensions()[0], g->inputDimensions()[1], imgDims[0], imgDims[1], imgDims[2]}, fft_.inputDimensions()}
    , apo_{pad_.inputDimensions(), g}
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
    tf_.resize(gridder_->inputDimensions());
    tf_.setConstant(1.f);
    tf_ = gridder_->Adj(gridder_->A(tf_));
  }

  template <typename T>
  auto A(T const &x) const
  {
    Log::Debug("Starting NUFFT forward");
    auto const &start = Log::Now();
    auto result = gridder_->A(fft_.A(pad_.A(apo_.A(x))));
    Log::Debug("Finished NUFFT forward: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    Log::Debug("Starting NUFFT adjoint");
    auto const start = Log::Now();
    auto result = apo_.Adj(pad_.Adj(fft_.Adj(gridder_->Adj(x))));
    Log::Debug("Finished NUFFT adjoint: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  auto AdjA(T const &x) const
  {
    if (tf_.size() == 0) {
      return Adj(A(x));
    } else {
      Log::Debug("Starting NUFFT Töplitz embedded adjoint*forward");
      auto const start = Log::Now();
      Eigen::IndexList<int, FixOne, FixOne, FixOne> brd;
      brd.set(0, this->inputDimensions()[0]);
      auto result = apo_.Adj(pad_.Adj(fft_.Adj(tf_.broadcast(brd) * fft_.A(pad_.A(apo_.A(x))))));
      Log::Debug("Finished NUFFT Töplitz embedded adjoint*forward: {}", Log::ToNow(start));
      return result;
    }
  }

private:
  GridBase *gridder_;
  FFTOp<5> fft_;
  PadOp<5> pad_;
  ApodizeOp<5> apo_;
  Cx5 tf_;
};
