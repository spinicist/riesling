#include "prox.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

template <typename S>
Prox<S>::Prox(Index const s)
  : sz{s}
{
}

template <typename S> void Prox<S>::primal(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->primal(α, xm, zm);
}

template <typename S> auto Prox<S>::primal(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->primal(α, x, z);
  return z;
}

template <typename S> void Prox<S>::dual(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->dual(α, xm, zm);
}

template <typename S> auto Prox<S>::dual(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->dual(α, x, z);
  return z;
}

template struct Prox<float>;
template struct Prox<Cx>;

} // namespace rl::Proxs
