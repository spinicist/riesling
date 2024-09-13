#include "norms.hpp"

#include "log.hpp"
#include "tensors.hpp"

namespace rl::Proxs {

L1::L1(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("Prox", "L1 / Soft Threshold λ {}", λ);
}

void L1::apply(float const α, CMap const &x, Map &z) const
{
  float t = α * λ;
  z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
  Log::Debug("Prox", "Soft Threshold α {} λ {} t {} |x| {} |z| {}", α, λ, t, x.stableNorm(), z.stableNorm());
}

void L1::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float t = λ * realα->scale;
    z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
    Log::Debug("Prox", "Soft Threshold λ {} t {} |x| {} |z| {}", λ, t, x.stableNorm(), z.stableNorm());
  } else {
    Log::Fail("Prox", "C++ is stupid");
  }
}

L2::L2(float const λ_, Index const sz_, Index const blk)
  : Prox<Cx>(sz_)
  , λ{λ_}
  , blockSize{blk}
{
  if (sz_ % blockSize != 0) { Log::Fail("Prox", "Block size {} does not cleanly divide {}", blockSize, sz_); }
  if (blockSize == 0) { blockSize = sz_; }
  Log::Print("Prox", "L2 Prox λ {} scaled λ {} block size {}", λ_, λ, blockSize);
}

void L2::apply(float const α, CMap const &x, Map &z) const
{
  float const t = α * λ;
  auto const  blks = x.rows() / blockSize;
  for (Index ib = 0; ib < blks; ib++) {
    auto const n = x.segment(ib * blockSize, blockSize).stableNorm();
    if (n > t) {
      z.segment(ib * blockSize, blockSize) = x.segment(ib * blockSize, blockSize) * (1.f - t / n);
    } else {
      z.segment(ib * blockSize, blockSize).setZero();
    }
  }
  Log::Debug("Prox", "L2 Prox α {} λ {} t {} |x| {} |z| {}", α, λ, t, x.stableNorm(), z.stableNorm());
}

void L2::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float      t = λ * realα->scale;
    auto const blks = x.rows() / blockSize;
    for (Index ib = 0; ib < blks; ib++) {
      auto const n = x.segment(ib * blockSize, blockSize).stableNorm();
      if (n > t) {
        z.segment(ib * blockSize, blockSize) = x.segment(ib * blockSize, blockSize) * (1.f - t / n);
      } else {
        z.segment(ib * blockSize, blockSize).setZero();
      }
    }
    Log::Debug("Prox", "L2 Prox λ {} t {} |x| {} |z| {}", λ, t, x.stableNorm(), z.stableNorm());
  } else {
    Log::Fail("Prox", "C++ is stupid");
  }
}

} // namespace rl::Proxs
