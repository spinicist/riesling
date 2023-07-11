#include "l2.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl::Proxs {

template <typename S>
L2<S>::L2(float const λ_, Index const sz_)
  : Prox<S>(sz_)
  , λ{λ_}
  , y{nullptr, sz_}
{
  Log::Print("L2 Prox λ {}", λ);
}

template <typename S>
L2<S>::L2(float const λ_, CMap const bias)
  : Prox<S>(bias.rows())
  , λ{λ_}
  , y{bias}
{
  Log::Print("L2 Prox λ {}", λ);
}

template <typename S>
void L2<S>::apply(float const α, CMap const &x, Map &z) const
{
  float const t = α * λ;
  Log::Print<Log::Level::High>("L2 λ {} starting", λ);
  if (y.data()) {
    z = (x - t * y) / (1.f + t);
  } else {
    z = x / (1.f + t);
  }
  Log::Print("L2 α {} λ {} t {} |x| {} |y| {} |z| {}", α, λ, t, x.norm(), y.norm(), z.norm());
}

template <typename S>
void L2<S>::apply(std::shared_ptr<Ops::Op<S>> const α, CMap const &x, Map &z) const
{
  auto const div = α->inverse(1.f, λ);
  Log::Print<Log::Level::High>("L2 λ {} starting", λ);
  if (y.data()) {
    z = div->forward(x - λ * α->forward(y));
  } else {
    z = div->forward(x);
  }
  Log::Print("L2 λ {} |x| {} |y| {} |z| {}", λ, x.norm(), y.norm(), z.norm());
}

template <typename S>
void L2<S>::setBias(S const *data)
{
  new (&this->y) CMap(data, this->sz);
}

template struct L2<float>;
template struct L2<Cx>;

} // namespace rl::Proxs