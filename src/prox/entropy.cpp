#include "entropy.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl::Proxs {

Entropy::Entropy(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("Entropy Prox λ {}", λ);
}

void Entropy::apply(float const α, CMap const &v, Map &z) const
{
  float const          t = α * λ;
  Eigen::ArrayXf const vabs = v.array().abs();
  Eigen::ArrayXf       x = vabs;
  for (int ii = 0; ii < 16; ii++) {
    auto const g = (x > 0.f).select((x.log() + 1.f) + (1.f / t) * (x - vabs), 0.f);
    x = (x - (t / 2.f) * g).cwiseMax(0.f);
  }
  z = v.array() * (x / vabs);
  Log::Print<Log::Level::High>("Entropy α {} λ {} t {} |v| {} |z| {}", α, λ, t, v.norm(), z.norm());
}

NMREntropy::NMREntropy(float const λ_, Index const sz_)
  : Prox<Cx>(sz_)
  , λ{λ_}
{
  Log::Print("NMR Entropy Prox λ {}", λ_);
}

void NMREntropy::apply(float const α, CMap const &v, Map &z) const
{
  float const          t = α * λ;
  Eigen::ArrayXf const vabs = v.array().abs();
  Eigen::ArrayXf       x = vabs;
  for (int ii = 0; ii < 16; ii++) {
    auto const xx = (x.square() + 1.f).sqrt();
    auto const g = ((x * (x / xx + 1.f)) / (x + xx) + (x + xx).log() - x / xx) + (1.f / t) * (x - vabs);
    x = (x - (t / 2.f) * g).cwiseMax(0.f);
  }
  z = v.array() * (x / vabs);
  Log::Print<Log::Level::High>("NMR Entropy α {} λ {} t {} |v| {} |z| {}", α, λ, t, v.norm(), z.norm());
}

} // namespace rl::Proxs
