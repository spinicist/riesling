#include "prox.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"

namespace rl::Proxs {

Prox::Prox(Index const s)
  : sz{s}
{
}

void Prox::apply(float const α, Vector &x) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Map xm(x.data(), sz);
  this->apply(α, xm);
}

void Prox::conj(float const α, Vector &x) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Map xm(x.data(), sz);
  this->conj(α, xm);
}

Conjugate::Conjugate(Prox::Ptr pp)
  : Prox{pp->sz}
  , p{pp}
{
}

auto Conjugate::Make(Prox::Ptr p) -> Prox::Ptr { return std::make_shared<Conjugate>(p); }

void Conjugate::apply(float const α, Map x) const
{
  Vector z = x / α;
  p->apply(1.f / α, z);
  x.device(Threads::CoreDevice()) = x - α * z;
}

void Conjugate::conj(float const α, Map x) const
{
  Vector z = x / α;
  p->conj(1.f / α, z);
  x.device(Threads::CoreDevice()) = x - α * z;
}

Null::Null(Index const isz)
  : Prox{isz}
{
}

auto Null::Make(Index const isz) -> Prox::Ptr { return std::make_shared<Null>(isz); }

void Null::apply(float const α, Map x) const { /* Null op */ }

void Null::conj(float const α, Map x) const { x.setZero(); }

} // namespace rl::Proxs
