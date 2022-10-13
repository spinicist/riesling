#include "thresh.hpp"

namespace rl {

SoftThreshold::SoftThreshold()
  : Prox<Cx4>()
{
}

auto SoftThreshold::operator()(float const λ, Cx4 const &x) const -> Cx4 const &
{
  Cx4 s = x * (x.abs() - λ) / x.abs();
  s = (x.abs() > λ).select(s, s.constant(0.f));
  return s;
}

} // namespace rl