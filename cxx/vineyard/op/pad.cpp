#include "pad.hpp"

#include "log.hpp"
#include "threads.hpp"

namespace rl::TOps {

template <typename S, int R>
Pad<S, R>::Pad(InDims const is, OutDims const os)
  : Parent(fmt::format("Pad {}D", InRank), is, os)
{
  for (Index ii = 0; ii < InRank; ii++) {
    if (ishape[ii] > oshape[ii]) { Log::Fail("Padding input dims {} larger than output dims {}", ishape, oshape); }
  }
  std::transform(oshape.begin(), oshape.end(), ishape.begin(), left_.begin(),
                 [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(oshape.begin(), oshape.end(), ishape.begin(), right_.begin(),
                 [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}

template <typename S, int R>
auto Pad<S, R>::inverse() const -> std::shared_ptr<rl::Ops::Op<Scalar>>
{
  return std::make_shared<Crop<Scalar, InRank>>(this->oshape, this->ishape);
}

template <typename S, int R> void Pad<S, R>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = x.pad(paddings_);
  this->finishForward(y, time, false);
}

template <typename S, int R> void Pad<S, R>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = y.slice(left_, ishape);
  this->finishAdjoint(x, time, false);
}

template <typename S, int R> void Pad<S, R>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::GlobalDevice()) += x.pad(paddings_);
  this->finishForward(y, time, true);
}

template <typename S, int R> void Pad<S, R>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::GlobalDevice()) += y.slice(left_, ishape);
  this->finishAdjoint(x, time, true);
}

template struct Pad<float, 1>;
template struct Pad<float, 2>;
template struct Pad<float, 3>;
template struct Pad<float, 4>;
template struct Pad<float, 5>;

template struct Pad<Cx, 1>;
template struct Pad<Cx, 2>;
template struct Pad<Cx, 3>;
template struct Pad<Cx, 4>;
template struct Pad<Cx, 5>;
template struct Pad<Cx, 6>;

template <typename S, int R>
Crop<S, R>::Crop(InDims const id, OutDims const od)
  : Parent(fmt::format("Crop {}D", InRank), id, od)
{
  for (Index ii = 0; ii < InRank; ii++) {
    if (ishape[ii] < oshape[ii]) { Log::Fail("Crop input dims {} smaller than output dims {}", ishape, oshape); }
  }
  std::transform(ishape.begin(), ishape.end(), oshape.begin(), left_.begin(),
                 [](Index big, Index small) { return (big - small + 1) / 2; });
  std::transform(ishape.begin(), ishape.end(), oshape.begin(), right_.begin(),
                 [](Index big, Index small) { return (big - small) / 2; });
  std::transform(left_.begin(), left_.end(), right_.begin(), paddings_.begin(),
                 [](Index left, Index right) { return std::make_pair(left, right); });
}
template <typename S, int R>
auto Crop<S, R>::inverse() const -> std::shared_ptr<rl::Ops::Op<Scalar>>
{
  return std::make_shared<Pad<Scalar, InRank>>(this->oshape, this->ishape);
}

template <typename S, int R> void Crop<S, R>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = x.slice(left_, oshape);
  this->finishForward(y, time, false);
}

template <typename S, int R> void Crop<S, R>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = y.pad(paddings_);
  this->finishAdjoint(x, time, false);
}

template <typename S, int R> void Crop<S, R>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::GlobalDevice()) += x.slice(left_, oshape);
  this->finishForward(y, time, true);
}

template <typename S, int R> void Crop<S, R>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::GlobalDevice()) += y.pad(paddings_);
  this->finishAdjoint(x, time, true);
}

template struct Crop<float, 1>;
template struct Crop<float, 2>;
template struct Crop<float, 3>;
template struct Crop<float, 4>;
template struct Crop<float, 5>;

template struct Crop<Cx, 1>;
template struct Crop<Cx, 2>;
template struct Crop<Cx, 3>;
template struct Crop<Cx, 4>;
template struct Crop<Cx, 5>;
template struct Crop<Cx, 6>;


} // namespace rl::TOps
