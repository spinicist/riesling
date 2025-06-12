#include "l1-wavelets.hpp"

namespace rl::Proxs {

L1Wavelets::L1Wavelets(float const λ, Sz5 const shape, Index const W, std::vector<Index> const dims)
  : Prox<Cx>(Product(shape))
  , waves_{std::make_shared<TOps::Wavelets<5>>(shape, W, dims)}
  , thresh_{λ, sz}
{
}

void L1Wavelets::apply(float const α, CMap x, Map z) const
{
  z = x;
  waves_->forward(z);
  CMap zm(z.data(), z.size());
  thresh_.apply(α, zm, z);
  waves_->adjoint(z);
}

} // namespace rl::Proxs
