#pragma once

#include "operator.hpp"

namespace rl {

template <int Rank>
struct PadOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;

  PadOp(InputDims const &inSize, OutputDims const &outSize)
  {
    for (Index ii = 0; ii < Rank; ii++) {
      if (outSize[ii] < inSize[ii]) {
        Log::Fail(FMT_STRING("Padding dim {}={} < input dim {}"), ii, outSize[ii], inSize[ii]);
      }
      input_[ii] = inSize[ii];
      output_[ii] = outSize[ii];
    }
    std::copy_n(inSize.begin(), Rank, input_.begin());
    std::copy_n(outSize.begin(), Rank, output_.begin());
    std::transform(output_.begin(), output_.end(), input_.begin(), left_.begin(), [](Index big, Index small) {
      return (big - small + 1) / 2;
    });
    std::transform(output_.begin(), output_.end(), input_.begin(), right_.begin(), [](Index big, Index small) {
      return (big - small) / 2;
    });
    std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(), [](Index left, Index right) {
      return std::make_pair(left, right);
    });
  }

  InputDims inputDimensions() const
  {
    return input_;
  }

  OutputDims outputDimensions() const
  {
    return output_;
  }

  template <typename T>
  auto A(T const &x) const
  {
    return x.pad(paddings_);
  }

  template <typename T>
  auto Adj(T const &x) const
  {
    return x.slice(left_, input_);
  }

private:
  InputDims input_, output_, left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;
};
} // namespace rl
