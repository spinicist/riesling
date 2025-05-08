#include "prox.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

template <typename S>
Prox<S>::Prox(Index const s)
  : sz{s}
{
}

template <typename S> void Prox<S>::apply(float const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->apply(α, xm, zm);
}

template <typename S> void Prox<S>::apply(std::shared_ptr<Op> const α, Vector const &x, Vector &z) const
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  if (z.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", z.size(), sz); }
  CMap xm(x.data(), sz);
  Map  zm(z.data(), sz);
  this->apply(α, xm, zm);
}

template <typename S> auto Prox<S>::apply(float const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->apply(α, x, z);
  return z;
}

template <typename S> void Prox<S>::apply(std::shared_ptr<Op> const, CMap const , Map ) const
{
  throw Log::Failure("Prox", "Not implemented");
}

template <typename S> auto Prox<S>::apply(std::shared_ptr<Op> const α, Vector const &x) const -> Vector
{
  if (x.size() != sz) { throw Log::Failure("Prox", "x size {} did not match {}", x.size(), sz); }
  Vector z(sz);
  this->apply(α, x, z);
  return z;
}

template struct Prox<float>;
template struct Prox<Cx>;

template <typename S>
ConjugateProx<S>::ConjugateProx(std::shared_ptr<Prox<S>> pp)
  : Prox<S>{pp->sz}
  , p{pp}
{
}

template <typename S> void ConjugateProx<S>::apply(float const α, CMap const x, Map z) const
{
  Vector x1 = x / α;
  CMap   x1m(x1.data(), x1.size());
  p->apply(1.f / α, x1m, z);
  z *= -α;
  z += x;
}

template <typename S> void ConjugateProx<S>::apply(std::shared_ptr<Op> const α, CMap const x, Map z) const
{
  auto   αinv = α->inverse();
  Vector x1 = αinv->forward(x);
  CMap   x1m(x1.data(), x1.size());
  p->apply(αinv, x1m, z);
  z = -α->forward(z);
  z += x;
}

template struct ConjugateProx<float>;
template struct ConjugateProx<Cx>;

} // namespace rl::Proxs
