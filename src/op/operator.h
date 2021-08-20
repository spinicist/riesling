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

template <int InRank, int OutRank>
struct Operator
{
  using Input = Eigen::Tensor<Cx, InRank>;
  using InputDims = typename Input::Dimensions;
  using Output = Eigen::Tensor<Cx, OutRank>;
  using OutputDims = typename Output::Dimensions;

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;
  virtual void AdjA(Input const &x, Input &y) const = 0;

  virtual InputDims inSize() const = 0;
  virtual OutputDims outSize() const = 0;

  void checkInputSize(Input const &x) const
  {
    auto const &sz = inSize();
    auto mm = std::mismatch(sz.begin(), sz.end(), x.dimensions().begin());
    if (mm.first != sz.end()) {
      Log::Fail(
          "Mismatched input dimension {}: {} vs {}",
          std::distance(sz.begin(), mm.first),
          *mm.first,
          *mm.second);
    }
  }

  void checkOutputSize(Output const &x) const
  {
    auto const &sz = outSize();
    auto mm = std::mismatch(sz.begin(), sz.end(), x.dimensions().begin());
    if (mm.first != sz.end()) {
      Log::Fail(
          "Mismatched output dimension {}: {} vs {}",
          std::distance(sz.begin(), mm.first),
          *mm.first,
          *mm.second);
    }
  }
};
