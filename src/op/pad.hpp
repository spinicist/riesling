#pragma once

#include "log.hpp"
#include "operator.hpp"
#include "tensorOps.hpp"

namespace rl {

template <int Rank, int ImgRank = 3>
struct PadOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  mutable Input x_;
  mutable Output y_;

  PadOp(
    Eigen::DSizes<Index, ImgRank> const &imgSize,
    Eigen::DSizes<Index, ImgRank> const &padSize,
    Eigen::DSizes<Index, Rank - ImgRank> const &otherSize = {})
  {
    int constexpr ImgStart = Rank - ImgRank;
    for (Index ii = 0; ii < ImgStart; ii++) {
      input_[ii] = otherSize[ii];
      output_[ii] = otherSize[ii];
    }
    for (Index ii = 0; ii < ImgRank; ii++) {
      if (padSize[ii] < imgSize[ii]) {
        Log::Fail(FMT_STRING("Padding dim {}={} < input dim {}"), ii, padSize[ii], imgSize[ii]);
      }
      input_[ii + ImgStart] = imgSize[ii];
      output_[ii + ImgStart] = padSize[ii];
    }
    std::transform(output_.begin(), output_.end(), input_.begin(), left_.begin(), [](Index big, Index small) {
      return (big - small + 1) / 2;
    });
    std::transform(
      output_.begin(), output_.end(), input_.begin(), right_.begin(), [](Index big, Index small) { return (big - small) / 2; });
    std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(), [](Index left, Index right) {
      return std::make_pair(left, right);
    });

    Log::Print("PadOp {}->{}", imgSize, padSize);
    x_.resize(input_);
    y_.resize(output_);
  }

  InputDims inputDimensions() const { return input_; }

  OutputDims outputDimensions() const { return output_; }

  auto forward(Input const &x) const -> Output const &
  {
    this->checkForward(x, "PadOp");
    y_ = x.pad(paddings_);
    LOG_DEBUG(FMT_STRING("Padding Forward Norm {}->{}"), Norm(x), Norm(y_));
    return y_;
  }

  auto adjoint(Output const &y) const -> Input const &
  {
    this->checkAdjoint(y, "PadOp");
    x_ = y.slice(left_, input_);
    LOG_DEBUG(FMT_STRING("Padding Adjoint Norm {}->{}"), Norm(y), Norm(x_));
    return x_;
  }

private:
  InputDims input_, output_, left_, right_;
  Eigen::array<std::pair<Index, Index>, Rank> paddings_;
  float scale_;
};
} // namespace rl
