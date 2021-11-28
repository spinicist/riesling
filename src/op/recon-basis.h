#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "gridBase.h"
#include "sense.h"

struct ReconBasisOp final : Operator<4, 3>
{
  ReconBasisOp(GridBase *gridder, Cx4 const &maps, Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  Sz3 dimensions() const;
  Sz3 outputDimensions() const;
  void calcToeplitz(Info const &info);

private:
  GridBase *gridder_;
  Cx5 mutable grid_;
  Cx5 transfer_;
  SenseOp sense_;
  R3 apo_;
  FFT::Planned<5, 3> fft_;
  Log log_;
};
