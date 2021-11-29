#pragma once

#include "operator.h"

struct SenseOp final : Operator<4, 5>
{
  SenseOp(Cx4 const &maps, typename Output::Dimensions const &fullSize);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  long channels() const;
  InputDims inputDimensions() const;
  OutputDims outputDimensions() const;

private:
  Cx4 const &maps_;
  Output::Dimensions full_, left_, size_, right_;
};
