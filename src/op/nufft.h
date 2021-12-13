#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "grid.h"
#include "pad.h"

struct NufftOp final : Operator<5, 3>
{
  NufftOp(GridBase *gridder, Index const nc, Index const ne, Eigen::Array3l const mat, Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;

private:
  GridBase *gridder_;
  Cx5 mutable grid_;
  FFT::Planned<5, 3> fft_;
  PadOp<5> pad_;
  R3 apo_;
  Log log_;
};
