#pragma once

#include "operator.h"

#include "../precond/sdc.hpp"
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
    Sz5 fullDims = gridder_->inputDimensions();
    fullDims[0] = 1;
    tf_.resize(fullDims);
    gridder_->workspace().slice(Sz5{0, 0, 0, 0, 0}, fullDims).setConstant(1.f);
    gridder_->Adj(gridder_->A(1), 1);
    tf_ = gridder_->workspace().slice(Sz5{0, 0, 0, 0, 0}, fullDims);
    Log::Debug(
      FMT_STRING("NUFFT: Calculated Töplitz. TF dimensions {}"), fmt::join(tf_.dimensions(), ","));
  }

  template <typename T>
  auto A(T const &x) const
  {
    Log::Debug("Starting NUFFT forward");
    auto const &start = Log::Now();
    fft_.workspace().device(Threads::GlobalDevice()) = pad_.A(apo_.A(x));
    fft_.A(); // FFT shares gridder->workspace()
    auto result = gridder_->A();
    Log::Debug("Finished NUFFT forward: {}", Log::ToNow(start));
    return result;
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    Log::Debug("Starting NUFFT adjoint");
    auto const start = Log::Now();
    gridder_->Adj(x);
    fft_.Adj(); // FFT and gridder share workspace
    auto result = apo_.Adj(pad_.Adj(fft_.workspace()));
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
      fft_.workspace().device(Threads::GlobalDevice()) = pad_.A(x);
      fft_.A();
      fft_.workspace().device(Threads::GlobalDevice()) = tf_.broadcast(brd) * fft_.workspace();
      fft_.Adj();
      result.device(Threads::GlobalDevice()) = pad_.Adj(fft_.workspace());
    }
    Log::Debug("Finished NUFFT adjoint*forward: {}", Log::ToNow(start));
    return result;
  }

private:
  GridBase *gridder_;
  FFTOp<5> fft_;
  PadOp<5> pad_;
  ApodizeOp<5> apo_;
  Cx5 tf_;
};
