#pragma once

#include "../threads.h"
#include "gridBase.hpp"
#include "operator.h"

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
    std::transform(
      output_.begin(), output_.end(), input_.begin(), left_.begin(), [](Index big, Index small) {
        return (big - small + 1) / 2;
      });
    std::transform(
      output_.begin(), output_.end(), input_.begin(), right_.begin(), [](Index big, Index small) {
        return (big - small) / 2;
      });
    std::transform(
      left_.begin(), left_.end(), right_.begin(), paddings_.begin(), [](Index left, Index right) {
        return std::make_pair(left, right);
      });

    for (Index ii = 0; ii < Rank - 3; ii++) {
      resApo_[ii] = 1;
      brdApo_[ii] = input_[ii];
    }
    for (Index ii = Rank - 3; ii < Rank; ii++) {
      resApo_[ii] = input_[ii];
      brdApo_[ii] = 1;
    }
  }

  InputDims inputDimensions() const
  {
    return input_;
  }

  OutputDims outputDimensions() const
  {
    return output_;
  }

  void setApodization(GridBase *gridder)
  {
    apo_ = gridder->apodization(Sz3{input_[Rank - 3], input_[Rank - 2], input_[Rank - 1]});
  }

  void resetApodization()
  {
    apo_ = R3();
  }

  void Adj(Output const &x, Input &y) const
  {
    for (auto ii = 0; ii < Rank; ii++) {
      assert(x.dimension(ii) == output_[ii]);
      assert(y.dimension(ii) == input_[ii]);
    }

    if (apo_.size()) {
      y.device(Threads::GlobalDevice()) =
        x.slice(left_, input_) / apo_.cast<Cx>().reshape(resApo_).broadcast(brdApo_);
    } else {
      y.device(Threads::GlobalDevice()) = x.slice(left_, input_);
    }
  }

  void A(Input const &x, Output &y) const
  {
    for (auto ii = 0; ii < Rank; ii++) {
      assert(x.dimension(ii) == input_[ii]);
      assert(y.dimension(ii) == output_[ii]);
    }

    if (apo_.size()) {
      y.device(Threads::GlobalDevice()) =
        (x / apo_.cast<Cx>().reshape(resApo_).broadcast(brdApo_)).pad(paddings_);
    } else {
      y.device(Threads::GlobalDevice()) = x.pad(paddings_);
    }
  }

  void AdjA(Input const &x, Input &y) const
  {
    for (auto ii = 0; ii < Rank; ii++) {
      assert(x.dimension(ii) == input_[ii]);
      assert(y.dimension(ii) == input_[ii]);
    }

    y.device(Threads::GlobalDevice()) = x;
  }

private:
  InputDims input_, output_, left_, right_, resApo_, brdApo_;
  R3 apo_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;
};
