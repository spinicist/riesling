#include "thresh.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl::Prox {

SoftThreshold::SoftThreshold(float const λ_, Index const sz)
  : Prox<Cx>(sz)
  , λ{λ_}
{
  Log::Print("Soft Threshold Prox λ {}", λ);
}

void SoftThreshold::apply(float const α, CMap const &x, Map &z) const
{
  float t = α * λ;
  z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
  Log::Print("Soft Threshold α {} λ {} t {} |x| {} |z| {}", α, λ, t, x.norm(), z.norm());
}

void SoftThreshold::apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const
{
  if (auto realα = std::dynamic_pointer_cast<Ops::DiagScale<Cx>>(α)) {
    float t = λ * realα->scale;
    z = x.cwiseAbs().cwiseTypedGreater(t).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
    Log::Print("Soft Threshold α {} λ {} t {} |x| {} |z| {}", α, λ, t, x.norm(), z.norm());
  } else {
    Log::Fail("C++ is stupid");
  }
}

} // namespace rl::Prox