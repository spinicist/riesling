#include "prox.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"

namespace rl::Proxs {

Prox::Prox(Index const s)
  : sz{s}
{
}

void Prox::apply(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->apply(α, xm, zm);
}

auto Prox::apply(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->apply(α, x, z);
  return z;
}

void Prox::conj(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->conj(α, xm, zm);
}

auto Prox::conj(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->conj(α, x, z);
  return z;
}

Conjugate::Conjugate(Prox::Ptr pp)
  : Prox{pp->sz}
  , p{pp}
{
}

auto Conjugate::Make(Prox::Ptr p) -> Prox::Ptr { return std::make_shared<Conjugate>(p); }

void Conjugate::apply(float const α, CMap x, Map z) const
{
  z.device(Threads::CoreDevice()) = x - α * p->apply(1.f / α, x / α);
}

void Conjugate::conj(float const α, CMap x, Map z) const { z.device(Threads::CoreDevice()) = x - α * p->conj(1.f / α, x / α); }

Null::Null(Index const isz)
  : Prox{isz}
{
}

auto Null::Make(Index const isz) -> Prox::Ptr { return std::make_shared<Null>(isz); }

void Null::apply(float const α, CMap x, Map z) const { z.device(Threads::CoreDevice()) = x; }

void Null::conj(float const α, CMap x, Map z) const { z.setZero(); }

} // namespace rl::Proxs
