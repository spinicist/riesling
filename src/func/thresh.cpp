#include "thresh.hpp"

namespace rl {

SoftThreshold::SoftThreshold(float l)
  : Functor<Cx4>()
  , λ{l}
{
}

auto SoftThreshold::operator()(Cx4 const &x) const -> Cx4
{
  Cx4 s = x * (x.abs() - λ) / x.abs();
  s = (s.abs() > λ).select(s, s.constant(0.f));
  return s;
}

} // namespace rl