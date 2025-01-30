#include "f0.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

f0Segment::f0Segment(Re3 const &f0in, std::vector<float> const &τin)
  : Parent("f0SegmentOp", AddFront(f0in.dimensions(), 1), AddFront(f0in.dimensions(), (Index)τin.size()))
  , f0{f0in}
{
  Index const N = τin.size();
  τ.resize(Sz1{N});
  for (Index ii = 0; ii < N; ii++) {
    τ(ii) = τin[ii] * Cx(0.f, 2.f * M_PI);
  }

  v0f123.set(0, N);
  f0v123.set(1, f0.dimension(0));
  f0v123.set(2, f0.dimension(1));
  f0v123.set(3, f0.dimension(2));
}

void f0Segment::forward(InCMap const x, OutMap y) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::TensorDevice()) =
    x.broadcast(v0f123) * (f0.reshape(f0v123).broadcast(v0f123).cast<Cx>() * τ.reshape(v0f123).broadcast(f0v123)).exp();
  finishForward(y, time, false);
}

void f0Segment::iforward(InCMap const x, OutMap y) const
{
  auto const time = startForward(x, y, true);
  y.device(Threads::TensorDevice()) +=
    x.broadcast(v0f123) * (f0.reshape(f0v123).broadcast(v0f123).cast<Cx>() * τ.reshape(v0f123).broadcast(f0v123)).exp();
  finishForward(y, time, true);
}

void f0Segment::adjoint(OutCMap const y, InMap x) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) =
    (y * (f0.reshape(f0v123).broadcast(v0f123).cast<Cx>() * τ.reshape(v0f123).broadcast(f0v123).conjugate()).exp()).sum(Sz1{0});
  finishAdjoint(x, time, false);
}

void f0Segment::iadjoint(OutCMap const y, InMap x) const
{
  auto const time = startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) +=
    (y * (f0.reshape(f0v123).broadcast(v0f123).cast<Cx>() * τ.reshape(v0f123).broadcast(f0v123).conjugate()).exp()).sum(Sz1{0});
  finishAdjoint(x, time, true);
}

} // namespace rl::TOps
