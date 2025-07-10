#include "top-id.hpp"

namespace rl::TOps {

template <int R> Identity<R>::Identity(Sz<R> dims)
  : Parent("Identity", dims, dims)
{
}

template <int R> void Identity<R>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = Parent::startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x * x.constant(s);
  Parent::finishForward(y, time, false);
}

template <int R> void Identity<R>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = Parent::startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y * y.constant(s);
  Parent::finishAdjoint(x, time, false);
}

template <int R> void Identity<R>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = Parent::startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x * x.constant(s);
  Parent::finishForward(y, time, true);
}

template <int R> void Identity<R>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = Parent::startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y * y.constant(s);
  Parent::finishAdjoint(x, time, true);
}

template struct Identity<4>;
template struct Identity<5>;

} // namespace rl::TOps
