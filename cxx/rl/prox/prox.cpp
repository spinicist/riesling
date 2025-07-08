#include "prox.hpp"

#include "../log/log.hpp"

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

} // namespace rl::Proxs
