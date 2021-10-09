#pragma once

#include "operator.h"

struct SenseOp final : Operator<3, 4>
{
  SenseOp(Output &maps, typename Output::Dimensions const &fullSize);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  long channels() const;
  Sz3 dimensions() const;
  Sz4 outputDimensions() const;

private:
  Output maps_;
  Output::Dimensions full_, left_, size_, right_;
};

struct SenseBasisOp final : Operator<4, 5>
{
  SenseBasisOp(Cx4 &maps, typename Output::Dimensions const &fullSize);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  long channels() const;
  Sz3 dimensions() const;
  Sz5 outputDimensions() const;

private:
  Cx4 maps_;
  Output::Dimensions full_, left_, size_, right_;
};
