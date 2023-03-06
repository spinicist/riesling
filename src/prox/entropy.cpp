#include "entropy.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

Entropy::Entropy(float const λ)
  : Prox<Cx4>()
  , λ_{λ}
{
  Log::Print(FMT_STRING("MaxEnt Prox λ {}"), λ);
}

auto Entropy::operator()(float const α, Eigen::TensorMap<Cx4 const> v) const -> Cx4
{
  float const t = α * λ_;
  Re4 const vabs = v.abs();
  Re4 x = vabs;
  for (int ii = 0; ii < 8; ii++) {
      x.device(Threads::GlobalDevice()) = (x - (t/2.f) * (x.log() + 1.f)).cwiseMax(0.f);
  }
  Cx4 const s = v * (x / vabs).cast<Cx>();
  Log::Print(FMT_STRING("Entropy α {} λ {} t {} |x| {} |s| {}"), α, λ_, t, Norm(x), Norm(s));
  return s;
}

NMREnt::NMREnt(float const λ)
  : Prox<Cx4>()
  , λ_{λ}
{
  Log::Print(FMT_STRING("NMR Entropy Prox λ {}"), λ);
}

auto NMREnt::operator()(float const α, Eigen::TensorMap<Cx4 const> v) const -> Cx4
{
  float const t = α * λ_;
  Re4 const vabs = v.abs();
  Re4 x = vabs;
  for (int ii = 0; ii < 8; ii++) {
      auto const &xx = (x.square() + 1.f).sqrt();
      x.device(Threads::GlobalDevice()) =
        (x - (t / 2.f) * ((x * (x / xx + 1.f)) / (x + xx) + (x + xx).log() - x / xx)).cwiseMax(0.f);
  }
  Cx4 const s = v * (x / vabs).cast<Cx>();
  Log::Print(FMT_STRING("NMR Entropy α {} λ {} t {} |x| {} |s| {}"), α, λ_, t, Norm(x), Norm(s));
  return s;
}

} // namespace rl