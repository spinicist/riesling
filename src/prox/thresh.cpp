#include "thresh.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

template<int ND>
SoftThreshold<ND>::SoftThreshold(float const λ_, Sz<ND> const s)
  : Prox<Cx>()
  , λ{λ_}
  , shape{s}
{
  Log::Print(FMT_STRING("Soft Threshold Prox λ {}"), λ);
}

template<int ND>
void SoftThreshold<ND>::operator()(float const α, Vector const &x, Vector &z) const
{
  float t = α * λ;
  z = x.cwiseAbs().cwiseTypedGreater(0.f).select(x.array() * (x.array().abs() - t) / x.array().abs(), 0.f);
  Log::Print(FMT_STRING("Soft Threshold α {} λ {} t {} |x| {} |z| {}"), α, λ, t, x.norm(), z.norm());
}

template struct SoftThreshold<4>;
template struct SoftThreshold<5>;

} // namespace rl