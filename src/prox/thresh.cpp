#include "thresh.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

SoftThreshold::SoftThreshold(float const λ_)
  : Prox<Cx>()
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

} // namespace rl