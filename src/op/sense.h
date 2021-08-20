#pragma once

#include "operator.h"

struct SenseOp final : Operator<3, 4>
{
  SenseOp(Output &maps, typename Output::Dimensions const &fullSize);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  Input::Dimensions inSize() const;
  Output::Dimensions outSize() const;

private:
  Output maps_;
  Output::Dimensions full_, left_, size_, right_;
};
