#pragma once

#include "../log.hpp"
#include "types.hpp"

/* Linear Operator
 *
 * This is my attempt at some kind of bastardized linear operator struct.
 * The key weirdness here is that the operators track the rank of their inputs/outputs. Technically
 * a linear operator should only be applied to vectors and matrices, but within this context all of
 * those vectors represent higher-rank tensors that are for the purposes of the operator treated as
 * a vector.
 *
 * Hence here we track the input/output ranks.
 */

namespace rl {

template <size_t InRank, size_t OutRank, typename Scalar = Cx>
struct Operator
{
  static const size_t InputRank = InRank;
  using Input = Eigen::Tensor<Scalar, InputRank>;
  using InputDims = typename Input::Dimensions;
  static const size_t OutputRank = OutRank;
  using Output = Eigen::Tensor<Scalar, OutputRank>;
  using OutputDims = typename Output::Dimensions;

  virtual ~Operator() {};

  virtual OutputDims outputDimensions() const = 0;
  virtual InputDims inputDimensions() const = 0;

  virtual auto forward(Input const &x) const -> Output const & = 0;
  virtual auto adjoint(Output const &y) const -> Input const & = 0;
  virtual auto adjfwd(Input const &x) const -> Input { Log::Fail("AdjFwd Not implemented"); }
}; // namespace rl

} // namespace rl
