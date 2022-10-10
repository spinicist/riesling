#pragma once

#include "operator.hpp"

namespace rl {

template <size_t Rank, typename Scalar = Cx>
struct Identity final : Operator<Rank, Rank, Scalar>
{
  using Parent = Operator<Rank, Rank, Scalar>;
  static const size_t InputRank = Parent::InputRank;
  using InputDims = typename Parent::InputDims;
  using Input = typename Parent::Input;
  static const size_t OutputRank = Parent::OutputRank;
  using OutputDims = typename Parent::OutputDims;
  using Output = typename Parent::Output;

  Identity(InputDims const d) : dims{d} {}
  InputDims dims;


  OutputDims outputDimensions() const { return dims; }
  InputDims inputDimensions() const { return dims; }

  virtual auto forward(Input const &x) const -> Output const & { return x; }
  virtual auto adjoint(Output const &y) const -> Input const & { return y; }

}; // namespace rl

} // namespace rl
