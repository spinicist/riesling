#pragma once

#include "gridBase.hpp"

namespace rl {

template<typename Scalar>
struct ApodizeOp final : Operator<5, 5, Scalar>
{
  using Parent = Operator<5, 5>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;

  ApodizeOp(InputDims const &inSize, GridBase<Scalar> *gridder);
  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output;
  auto adjoint(Output const &x) const -> Input;

private:
  InputDims sz_, res_, brd_;
  Re3 apo_;
};

} // namespace rl
