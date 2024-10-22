#include "pad.hpp"

#include "log.hpp"
#include "op/top-impl.hpp"
#include "sys/threads.hpp"

#include "fmt/std.h"

namespace rl::TOps {

template <typename Scalar, int Rank>
Pad<Scalar, Rank>::Pad(InDims const is, OutDims const os)
  : Parent(fmt::format("Pad {}D", Rank), is, os)
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

template <typename Scalar, int Rank> void Pad<Scalar, Rank>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x.pad(paddings_);
  this->finishForward(y, time, false);
}

template <typename Scalar, int Rank> void Pad<Scalar, Rank>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  fmt::print(stderr, "|x| {} |y| {}\n", Norm(x), Norm(y));
  x = y.slice(left_, ishape);
  fmt::print(stderr, "Pad adjoint y {} left {} ishape {} x {}\n", y.dimensions(), left_, ishape, x.dimensions());
  fmt::print(stderr, "x {}\n", x.data()[0]);
  this->finishAdjoint(x, time, false);
}

template <typename Scalar, int Rank> void Pad<Scalar, Rank>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x.pad(paddings_);
  this->finishForward(y, time, true);
}

template <typename Scalar, int Rank> void Pad<Scalar, Rank>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y.slice(left_, ishape);
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

} // namespace rl::TOps
