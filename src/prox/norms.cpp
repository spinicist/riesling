#include "norms.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl::Proxs {

L1::L1(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("L1 / Soft Threshold Prox. λ {}", λ);
}

void L1::apply(float const α, CMap const &x, Map &z) const
{
  float t = α * λ;
  z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
  Log::Print<Log::Level::High>("Soft Threshold α {} λ {} t {} |x| {} |z| {}", α, λ, t, x.norm(), z.norm());
}

void L1::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float t = λ * realα->scale;
    z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
    Log::Print<Log::Level::High>("Soft Threshold λ {} t {} |x| {} |z| {}", λ, t, x.norm(), z.norm());
  } else {
    Log::Fail("C++ is stupid");
  }
}

L2::L2(float const λ_, Index const sz_, Index const blk)
  : Prox<Cx>(sz_)
  , λ{λ_}
  , block{blk}
{
  Log::Print("L2 Prox λ {}", λ);
  if (sz_ % block != 0) { Log::Fail("Block size {} does not cleanly divide {}", block, sz_); }
  if (block == 0) { block = sz_; }
}

void L2::apply(float const α, CMap const &x, Map &z) const
{
  float const t = α * λ;
  auto const  norms =
    x.reshaped(block, x.rows() / block).colwise().norm().replicate(block, x.rows() / block).reshaped(x.rows(), 1).array();
  z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (1.f - t / norms), 0.f);
  Log::Print<Log::Level::High>("L2 Prox α {} λ {} t {} |x| {} |z| {}", α, λ, t, x.norm(), z.norm());
}

void L2::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float      t = λ * realα->scale;
    auto const norms =
      x.reshaped(block, x.rows() / block).colwise().norm().replicate(block, x.rows() / block).reshaped(x.rows(), 1).array();
    z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (1.f - t / norms), 0.f);
    Log::Print<Log::Level::High>("L2 Prox λ {} t {} |x| {} |z| {}", λ, t, x.norm(), z.norm());
  } else {
    Log::Fail("C++ is stupid");
  }
}

} // namespace rl::Proxs