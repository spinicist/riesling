#include "crop.h"

#include <algorithm>

#include "../threads.h"

template <int Rank>
CropOp<Rank>::CropOp(InputDims const &bigSize, OutputDims const &smallSize)
{
  std::copy_n(bigSize.begin(), Rank, full_.begin());
  std::copy_n(smallSize.begin(), Rank, size_.begin());
  std::transform(
      bigSize.begin(), bigSize.end(), smallSize.begin(), left_.begin(), [](long big, long small) {
        return (big - small + 1) / 2;
      });
  std::transform(
      bigSize.begin(), bigSize.end(), smallSize.begin(), right_.begin(), [](long big, long small) {
        return (big - small) / 2;
      });
}

template <int Rank>
typename CropOp<Rank>::InputDims CropOp<Rank>::bigDimensions() const
{
  return full_;
}

template <int Rank>
typename CropOp<Rank>::OutputDims CropOp<Rank>::outputDimensions() const
{
  return size_;
}

template <int Rank>
void CropOp<Rank>::A(Input const &x, Output &y) const
{
  for (auto ii = 0; ii < Rank; ii++) {
    assert(x.dimension(ii) == full_[ii]);
    assert(y.dimension(ii) == size_[ii]);
  }
  y.device(Threads::GlobalDevice()) = x.slice(left_, size_);
}

template <int Rank>
void CropOp<Rank>::Adj(Output const &x, Input &y) const
{
  for (auto ii = 0; ii < Rank; ii++) {
    assert(x.dimension(ii) == size_[ii]);
    assert(y.dimension(ii) == full_[ii]);
  }
  Eigen::array<std::pair<int, int>, Rank> paddings;
  std::transform(
      left_.begin(), left_.end(), right_.begin(), paddings.begin(), [](long left, long right) {
        return std::make_pair(left, right);
      });
  y.device(Threads::GlobalDevice()) = x.pad(paddings);
}

template <int Rank>
void CropOp<Rank>::AdjA(Input const &x, Input &y) const
{
  for (auto ii = 0; ii < Rank; ii++) {
    assert(x.dimension(ii) == full_[ii]);
    assert(y.dimension(ii) == full_[ii]);
  }
  Eigen::array<std::pair<int, int>, Rank> paddings;
  std::transform(
      left_.begin(), left_.end(), right_.begin(), paddings.begin(), [](long left, long right) {
        return std::make_pair(left, right);
      });
  y.device(Threads::GlobalDevice()) = x.slice(left_, size_).pad(paddings);
}

template struct CropOp<3>;
template struct CropOp<4>;
