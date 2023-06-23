#include "l2.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl::Proxs {

L2::L2(float const λ_, CMap const bias)
  : Prox<Cx>(bias.rows())
  , λ{λ_}
  , y{bias}
{
  Log::Print("L2 Prox λ {}", λ);
}

void L2::apply(float const α, CMap const &x, Map &z) const
{
  float const t = α * λ;
  z = (x - t * y) / (1.f + t);
  Log::Print("L2 α {} λ {} t {} |x| {} |y| {} |z| {}", α, λ, t, x.norm(), y.norm(), z.norm());
}

void L2::apply(std::shared_ptr<Ops::Op<Cx>> const α, CMap const &x, Map &z) const
{
  auto const div = α->inverse(1.f, λ);
  z = div->forward(x - λ * α->forward(y));
  Log::Print("L2 λ {} |x| {} |y| {} |z| {}", λ, x.norm(), y.norm(), z.norm());
}

} // namespace rl::Proxs