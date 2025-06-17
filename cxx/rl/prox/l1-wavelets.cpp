#include "l1-wavelets.hpp"

namespace rl::Proxs {

L1Wavelets::L1Wavelets(float const λ, Sz5 const shape, Index const W, std::vector<Index> const dims)
  : Prox<Cx>(Product(shape))
  , waves{std::make_shared<TOps::Wavelets<5>>(shape, W, dims)}
  , l1{λ, sz}
{
}

void L1Wavelets::primal(float const α, CMap x, Map z) const
{
  waves->forward(x, z);
  CMap zm(z.data(), z.size());
  l1.primal(α, zm, z);
  CMap zc(z.data(), z.size());
  waves->adjoint(zc, z);
}

void L1Wavelets::dual(float const α, CMap x, Map z) const
{
  waves->forward(x, z);
  CMap zm(z.data(), z.size());
  l1.dual(α, zm, z);
  CMap zc(z.data(), z.size());
  waves->adjoint(zc, z);
}

} // namespace rl::Proxs
