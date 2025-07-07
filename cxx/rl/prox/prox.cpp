#include "prox.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

Prox::Prox(Index const s)
  : sz{s}
{
}

void Prox::primal(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->primal(α, xm, zm);
}

auto Prox::primal(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->primal(α, x, z);
  return z;
}

void Prox::dual(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->dual(α, xm, zm);
}

auto Prox::dual(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->dual(α, x, z);
  return z;
}

} // namespace rl::Proxs
