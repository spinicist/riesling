#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "grid.h"
#include "sense.h"

struct ReconOp final : Operator<3, 3>
{
  ReconOp(
      Trajectory const &traj,
      float const os,
      bool const kb,
      bool const fast,
      std::string const sdc,
      Cx4 &maps,
      Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  Sz3 dimensions() const;
  Sz3 outputDimensions() const;
  void setPreconditioning(float const p);
  void calcToeplitz(Info const &info);

private:
  std::unique_ptr<GridOp> gridder_;
  Cx4 mutable grid_;
  Cx4 transfer_;
  SenseOp sense_;
  R3 apo_;
  FFT::ThreeDMulti fft_;
  Log log_;
};
