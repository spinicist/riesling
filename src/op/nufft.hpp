#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "grid.h"
#include "pad.hpp"

struct NufftOp final : Operator<5, 3>
{

  NufftOp(GridBase *gridder, Index const nc, Index const ne, Eigen::Array3l const mat)
    : gridder_{gridder}
    , grid_{Sz5{
        nc,
        ne,
        gridder_->mapping().cartDims[0],
        gridder_->mapping().cartDims[1],
        gridder_->mapping().cartDims[2]}}
    , fft_{grid_}
    , pad_{Sz5{nc, ne, mat[0], mat[1], mat[2]}, grid_.dimensions()}

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

  void A(Input const &x, Output &y) const
  {
    auto const &start = Log::Now();

    pad_.A(x, grid_); // Pad takes care of apodization
    fft_.forward(grid_);
    gridder_->A(grid_, y);
    Log::Debug("Encode: {}", Log::ToNow(start));
  }

  void Adj(Output const &x, Input &y) const
  {
    auto const &start = Log::Now();
    gridder_->Adj(x, grid_);
    fft_.reverse(grid_);
    pad_.Adj(grid_, y); // This is a crop which takes care of apodization
    Log::Debug("Decode: {}", Log::ToNow(start));
  }

  void AdjA(Input const &x, Input &y) const
  {
    Output temp(gridder_->mapping().noncartDims);
    A(x, temp);
    Adj(temp, y);
  }

private:
  GridBase *gridder_;
  Cx5 mutable grid_;
  FFT::Planned<5, 3> fft_;
  PadOp<5> pad_;
  R3 apo_;
};
