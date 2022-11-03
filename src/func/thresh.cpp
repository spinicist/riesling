#include "thresh.hpp"

namespace rl {

SoftThreshold::SoftThreshold()
  : Prox<Cx4>()
{
}

auto SoftThreshold::operator()(float const λ, Eigen::TensorMap<Cx4 const>x) const -> Cx4
{
  Cx4 s = (x.abs() > λ).select(x * (x.abs() - λ) / x.abs(), x.constant(0.f));
  return s;
}

} // namespace rl