#pragma once

#include "../log.h"
#include "gridBase.h"
#include "operator.h"

struct SenseOp final : Operator<4, 5>
{
  SenseOp(Cx4 const &maps, typename Output::Dimensions const &fullSize, Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  Index channels() const;
  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;
  void setApodization(GridBase *gridder); // Combine apodization into SENSE to save memory
  void resetApodization();

private:
  Cx4 const &maps_;
  Log log_;
  R3 apo_;
  Output::Dimensions full_, left_, size_, right_;
};
