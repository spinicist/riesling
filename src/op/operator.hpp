#pragma once

#include "../log.h"
#include "../types.h"

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

template <int InRank, int OutRank, typename Scalar = Cx>
struct Operator
{
  using Input = Eigen::Tensor<Scalar, InRank>;
  using InputDims = typename Input::Dimensions;
  using Output = Eigen::Tensor<Scalar, OutRank>;
  using OutputDims = typename Output::Dimensions;

  virtual OutputDims outputDimensions() const = 0;
  virtual InputDims inputDimensions() const = 0;
};
