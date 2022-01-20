#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "grid.h"
#include "sense.hpp"

struct ReconOp final : Operator<4, 3>
{
  ReconOp(GridBase *gridder, Cx4 const &maps)
    : gridder_{gridder}
    , grid_{gridder_->inputDimensions(maps.dimension(0))}
    , sense_{maps, grid_.dimensions()}
    , fft_{grid_}

  {
    sense_.setApodization(gridder);
  }

  InputDims inputDimensions() const
  {
    return sense_.inputDimensions();
  }

  OutputDims outputDimensions() const
  {
    return gridder_->outputDimensions();
  }

  void calcToeplitz(Info const &info)
  {
    Log::Print("Calculating Töplitz embedding");
    transfer_.resize(gridder_->inputDimensions(1));
    transfer_.setConstant(1.f);
    Cx3 tf(1, info.read_points, info.spokes);
    gridder_->A(transfer_, tf);
    gridder_->Adj(tf, transfer_);
    sense_.resetApodization();
  }

  void A(Input const &x, Output &y) const
  {
    auto const &start = Log::Now();
    sense_.A(x, grid_); // SENSE takes care of apodization
    fft_.forward(grid_);
    gridder_->A(grid_, y);
    Log::Debug("Encode: {}", Log::ToNow(start));
  }

  void Adj(Output const &x, Input &y) const
  {
    auto const &start = Log::Now();
    gridder_->Adj(x, grid_);
    fft_.reverse(grid_);
    sense_.Adj(grid_, y); // SENSE takes care of apodization
    Log::Debug("Decode: {}", Log::ToNow(start));
  }

  void AdjA(Input const &x, Input &y) const
  {
    if (transfer_.size() == 0) {
      Output temp(outputDimensions());
      A(x, temp);
      Adj(temp, y);
    } else {
      auto dev = Threads::GlobalDevice();
      auto const start = Log::Now();
      sense_.A(x, grid_);
      fft_.forward(grid_);
      Eigen::IndexList<int, FixOne, FixOne, FixOne> brd;
      brd.set(0, grid_.dimension(0));
      grid_.device(dev) = grid_ * transfer_.broadcast(brd);
      fft_.reverse(grid_);
      sense_.Adj(grid_, y);
      Log::Debug("Töplitz embedded: {}", Log::ToNow(start));
    }
  }

private:
  GridBase *gridder_;
  Cx5 mutable grid_;
  Cx5 transfer_;
  SenseOp sense_;
  FFT::Planned<5, 3> fft_;
};
