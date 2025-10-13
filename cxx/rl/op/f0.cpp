#include "f0.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

f0Segment::f0Segment(Re3 const &f0in, float const τ0, float const τacq, Index const Nτ, Index const Nacq)
  : Parent("f0SegmentOp", AddBack(f0in.dimensions(), 1), AddBack(f0in.dimensions(), Nτ))
  , f0{f0in}
  , τ(Nτ)
{
  if (τacq == 0.f) { throw Log::Failure("f0", "τacq must be non-zero for f0 correction"); }
  if (Nτ < 2) { throw Log::Failure("f0", "Must have more than 1 time segment for f0 correction"); }
  float const dτ = τacq / (Nτ - 1);
  Index const N = 2 * Nacq / (Nτ - 1);
  Cx3         basis(Nτ, Nacq, 1);
  basis.setZero();
  for (Index ii = 0; ii < Nτ; ii++) {
    τ(ii) = -(τ0 + ii * dτ) * Cx(0.f, 2.f * M_PI);

    Index const start = ii * N / 2 - N / 2;
    for (Index ij = 0; ij < N; ij++) {
      Index const io = start + ij;
      if (io >= 0 && io < Nacq) { basis(ii, io, 0) = std::pow(std::sin(M_PI * ij / N), 2); }
    }
  }
  b = Basis(basis);
  f012v3.set(3, Nτ);
  v012f3.set(0, f0.dimension(0));
  v012f3.set(1, f0.dimension(1));
  v012f3.set(2, f0.dimension(2));
}

auto f0Segment::basis() const -> Basis::CPtr { return &this->b; }

void f0Segment::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = startForward(x, y, false);
  y.device(Threads::TensorDevice()) =
    x.broadcast(f012v3) * (f0.reshape(v012f3).broadcast(f012v3).cast<Cx>() * τ.reshape(f012v3).broadcast(v012f3)).exp() *
    y.constant(s);
  finishForward(y, time, false);
}

void f0Segment::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = startForward(x, y, true);
  y.device(Threads::TensorDevice()) +=
    x.broadcast(f012v3) * (f0.reshape(v012f3).broadcast(f012v3).cast<Cx>() * τ.reshape(f012v3).broadcast(v012f3)).exp() *
    y.constant(s);
  finishForward(y, time, true);
}

void f0Segment::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) =
    (y * (f0.reshape(v012f3).broadcast(f012v3).cast<Cx>() * τ.reshape(f012v3).broadcast(v012f3)).exp().conjugate())
      .sum(Sz1{3})
      .reshape(v012f3) *
    x.constant(s);
  finishAdjoint(x, time, false);
}

void f0Segment::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = startAdjoint(y, x, true);
  x.device(Threads::TensorDevice()) +=
    (y * (f0.reshape(v012f3).broadcast(f012v3).cast<Cx>() * τ.reshape(f012v3).broadcast(v012f3)).exp().conjugate())
      .sum(Sz1{3})
      .reshape(v012f3) *
    x.constant(s);
  finishAdjoint(x, time, true);
}

} // namespace rl::TOps
