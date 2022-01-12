#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "grid.h"
#include "sense.hpp"

struct ReconOp final : Operator<4, 3>
{
  ReconOp(GridBase *gridder, Cx4 const &maps, Log &log)
    : gridder_{gridder}
    , grid_{gridder_->inputDimensions(maps.dimension(0))}
    , sense_{maps, grid_.dimensions(), log}
    , fft_{grid_, log}
    , log_{log}
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
    log_.info("Calculating Töplitz embedding");
    transfer_.resize(gridder_->inputDimensions(1));
    transfer_.setConstant(1.f);
    Cx3 tf(1, info.read_points, info.spokes);
    gridder_->A(transfer_, tf);
    gridder_->Adj(tf, transfer_);
    sense_.resetApodization();
  }

  void A(Input const &x, Output &y) const
  {
    auto const &start = log_.now();
    sense_.A(x, grid_); // SENSE takes care of apodization
    fft_.forward(grid_);
    gridder_->A(grid_, y);
    log_.debug("Encode: {}", log_.toNow(start));
  }

  void Adj(Output const &x, Input &y) const
  {
    auto const &start = log_.now();
    gridder_->Adj(x, grid_);
    fft_.reverse(grid_);
    sense_.Adj(grid_, y); // SENSE takes care of apodization
    log_.debug("Decode: {}", log_.toNow(start));
  }

  void AdjA(Input const &x, Input &y) const
  {
    if (transfer_.size() == 0) {
      Output temp(outputDimensions());
      A(x, temp);
      Adj(temp, y);
    } else {
      auto dev = Threads::GlobalDevice();
      auto const start = log_.now();
      sense_.A(x, grid_);
      fft_.forward(grid_);
      Eigen::IndexList<int, FixOne, FixOne, FixOne> brd;
      brd.set(0, grid_.dimension(0));
      grid_.device(dev) = grid_ * transfer_.broadcast(brd);
      fft_.reverse(grid_);
      sense_.Adj(grid_, y);
      log_.debug("Töplitz embedded: {}", log_.toNow(start));
    }
  }

private:
  GridBase *gridder_;
  Cx5 mutable grid_;
  Cx5 transfer_;
  SenseOp sense_;
  FFT::Planned<5, 3> fft_;
  Log log_;
};
