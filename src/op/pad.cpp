#include "pad.h"

#include <algorithm>

#include "../log.h"
#include "../threads.h"

template <int Rank>
PadOp<Rank>::PadOp(InputDims const &inSize, OutputDims const &outSize)
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

template <int Rank>
typename PadOp<Rank>::InputDims PadOp<Rank>::inputDimensions() const
{
  return input_;
}

template <int Rank>
typename PadOp<Rank>::OutputDims PadOp<Rank>::outputDimensions() const
{
  return output_;
}

template <int Rank>
void PadOp<Rank>::setApodization(GridBase *gridder)
{
  apo_ = gridder->apodization(Sz3{input_[Rank - 3], input_[Rank - 2], input_[Rank - 1]});
}

template <int Rank>
void PadOp<Rank>::resetApodization()
{
  apo_ = R3();
}

template <int Rank>
void PadOp<Rank>::Adj(Output const &x, Input &y) const
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

template <int Rank>
void PadOp<Rank>::A(Input const &x, Output &y) const
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

template <int Rank>
void PadOp<Rank>::AdjA(Input const &x, Input &y) const
{
  for (auto ii = 0; ii < Rank; ii++) {
    assert(x.dimension(ii) == input_[ii]);
    assert(y.dimension(ii) == input_[ii]);
  }

  y.device(Threads::GlobalDevice()) = x;
}

template struct PadOp<3>;
template struct PadOp<4>;
template struct PadOp<5>;
