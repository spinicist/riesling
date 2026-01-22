#include "pad.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int Rank> Pad<Rank>::Pad(InDims const is, OutDims const os, float const scale)
  : Parent(fmt::format("Pad {}D", Rank), is, os)
  , scale_(scale)
{
  for (Index ii = 0; ii < Rank; ii++) {
    if (ishape[ii] > oshape[ii]) {
      throw Log::Failure("Pad", "Padding input dims {} larger than output dims {}", ishape, oshape);
    }
  }
  std::transform(oshape.begin(), oshape.end(), ishape.begin(), left_.begin(),
                 [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(oshape.begin(), oshape.end(), ishape.begin(), right_.begin(),
                 [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int Rank> auto Pad<Rank>::Make(InDims const is, OutDims const os, float const scale) -> Ptr
{
  return std::make_shared<Pad<Rank>>(is, os, scale);
}

template <int Rank> void Pad<Rank>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.pad(paddings_) * y.constant(s * scale_);
  this->finishForward(y, time, false);
}

template <int Rank> void Pad<Rank>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y.slice(left_, ishape) * x.constant(s / scale_);
  this->finishAdjoint(x, time, false);
}

template <int Rank> void Pad<Rank>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.pad(paddings_) * y.constant(s * scale_);
  this->finishForward(y, time, true);
}

template <int Rank> void Pad<Rank>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y.slice(left_, ishape) * x.constant(s / scale_);
  this->finishAdjoint(x, time, true);
}

template struct Pad<1>;
template struct Pad<2>;
template struct Pad<3>;
template struct Pad<4>;
template struct Pad<5>;
template struct Pad<6>;

template <int Rank> Crop<Rank>::Crop(InDims const is, OutDims const os, float const scale)
  : Parent(fmt::format("Crop {}D", Rank), is, os)
  , scale_(scale)
{
  for (Index ii = 0; ii < Rank; ii++) {
    if (ishape[ii] < oshape[ii]) {
      throw Log::Failure("Crop", "Cropping input dims {} smaller than output dims {}", ishape, oshape);
    }
  }
  std::transform(ishape.begin(), ishape.end(), oshape.begin(), left_.begin(),
                 [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(ishape.begin(), ishape.end(), oshape.begin(), right_.begin(),
                 [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <int Rank> auto Crop<Rank>::Make(InDims const is, OutDims const os, float const scale) -> Ptr
{
  return std::make_shared<Crop<Rank>>(is, os, scale);
}

template <int Rank> void Crop<Rank>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.slice(left_, oshape) * y.constant(s * scale_);
  this->finishForward(y, time, false);
}

template <int Rank> void Crop<Rank>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y.pad(paddings_) * x.constant(s / scale_);
  this->finishAdjoint(x, time, false);
}

template <int Rank> void Crop<Rank>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.slice(left_, oshape) * y.constant(s * scale_);
  this->finishForward(y, time, true);
}

template <int Rank> void Crop<Rank>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y.pad(paddings_) * x.constant(s / scale_);
  this->finishAdjoint(x, time, true);
}

template struct Crop<4>;
template struct Crop<5>;
template struct Crop<6>;

} // namespace rl::TOps
