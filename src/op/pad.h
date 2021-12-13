#pragma once

#include "gridBase.h"
#include "operator.h"

template <int Rank>
struct PadOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;

  PadOp(InputDims const &bigSize, OutputDims const &smallSize);
  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;
  void setApodization(GridBase *gridder); // Combine apodization into SENSE to save memory
  void resetApodization();

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

private:
  InputDims input_, output_, left_, right_, resApo_, brdApo_;
  R3 apo_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;
};
