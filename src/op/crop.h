#pragma once

#include "operator.h"

template <int Rank>
struct CropOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;

  CropOp(InputDims const &bigSize, OutputDims const &smallSize);
  InputDims bigDimensions() const;
  OutputDims outputDimensions() const;

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

private:
  InputDims full_, left_, size_, right_;
};

using CropOp3 = CropOp<3>;
using CropOp4 = CropOp<4>;
