#pragma once

#include "operator.h"

#include "../fft_plan.h"
#include "grid-basis.h"
#include "sense.h"

struct ReconBasisOp final : Operator<4, 3>
{
  ReconBasisOp(
      Trajectory const &traj,
      float const os,
      bool const kb,
      bool const fast,
      std::string const sdc,
      Cx4 &maps,
      R2 &basis,
      Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;

  Sz3 dimensions() const;
  Sz3 outputDimensions() const;
  void setPreconditioning(float const p);

private:
  std::unique_ptr<GridBasisOp> gridder_;
  Cx5 mutable grid_;
  SenseBasisOp sense_;
  R3 apo_;
  FFT::ThreeDBasis fft_;
  Log log_;
};