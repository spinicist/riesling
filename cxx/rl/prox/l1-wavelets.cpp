#include "l1-wavelets.hpp"

namespace rl::Proxs {

L1Wavelets::L1Wavelets(float const λ, Sz5 const shape, Index const W, std::vector<Index> const dims)
  : Prox(Product(shape))
  , waves{std::make_shared<TOps::Wavelets<5>>(shape, W, dims)}
  , l1{λ, sz}
{
}

void L1Wavelets::apply(float const α, Map x) const
{
  Vector z(x.size());
  waves->forward(x, z);
  l1.apply(α, z);
  CMap zm(z.data(), z.size());
  waves->adjoint(zm, x);
}

void L1Wavelets::conj(float const α, Map x) const
{
  Vector z(x.size());
  waves->forward(x, z);
  l1.conj(α, z);
  CMap zm(z.data(), z.size());
  waves->adjoint(zm, x);
}

} // namespace rl::Proxs
