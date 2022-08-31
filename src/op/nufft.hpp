#pragma once

#include "operator.hpp"

#include "apodize.hpp"
#include "fft.hpp"
#include "gridBase.hpp"
#include "pad.hpp"
#include "sdc.hpp"

namespace rl {

struct NUFFTOp final : Operator<5, 3>
{
  NUFFTOp(Sz3 const imgDims, GridBase<Cx, 3> *g, SDCOp *sdc = nullptr);

  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output;
  auto adjoint(Output const &x) const -> Input;
  auto adjfwd(Input const &x) const -> Input;
  auto fft() const -> FFTOp<5> const &;
  void calcToeplitz();

private:
  GridBase<Cx, 3> *gridder_;
  FFTOp<5> fft_;
  PadOp<5> pad_;
  ApodizeOp<Cx> apo_;
  Cx5 tf_;
  SDCOp *sdc_;
  float scale_;
};

} // namespace rl
