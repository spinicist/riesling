#include "top-id.hpp"

namespace rl::TOps {

template <typename S, int R>
Identity<S, R>::Identity(Sz<R> dims)
  : Parent("Identity", dims, dims)
{
}

template <typename S, int R> void Identity<S, R>::forward(InCMap x, OutMap y) const
{
  auto const time = Parent::startForward(x, y, false);
  y.device(Threads::TensorDevice()) = x;
  Parent::finishAdjoint(y, time, false);
}

template <typename S, int R> void Identity<S, R>::adjoint(OutCMap y, InMap x) const
{
  auto const time = Parent::startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = y;
  Parent::finishAdjoint(x, time, false);
}

template <typename S, int R> void Identity<S, R>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = Parent::startForward(x, y, true);
  y.device(Threads::TensorDevice()) += x * x.constant(s);
  Parent::finishAdjoint(y, time, true);
}

template <typename S, int R> void Identity<S, R>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = Parent::startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) += y * y.constant(s);
  Parent::finishAdjoint(x, time, true);
}

template struct Identity<Cx, 4>;
template struct Identity<Cx, 5>;

}
