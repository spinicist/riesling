#pragma once

#include "operator.hpp"
#include "gridBase.hpp"

namespace rl {

template<typename Scalar, size_t NDim>
struct ApodizeOp final : Operator<NDim + 2, NDim + 2, Scalar>
{
  using Parent = Operator<NDim + 2, NDim + 2>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  mutable Input x_;
  mutable Output y_;

  ApodizeOp(InputDims const &inSize, GridBase<Scalar, NDim> *gridder);
  auto inputDimensions() const -> InputDims;
  auto outputDimensions() const -> OutputDims;
  auto forward(Input const &x) const -> Output const &;
  auto adjoint(Output const &x) const -> Input const &;

private:
  InputDims sz_, res_, brd_;
  Eigen::Tensor<float, NDim> apo_;
};

} // namespace rl
