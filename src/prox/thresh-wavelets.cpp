#include "thresh-wavelets.hpp"

namespace rl {

ThresholdWavelets::ThresholdWavelets(float const λ, Sz4 const shape, Index const W, Index const L)
  : Prox<Cx>(Product(shape))
  , waves_{std::make_shared<Wavelets>(shape, W, L)}
  , thresh_{λ, sz}
{
}

void ThresholdWavelets::apply(float const α, CMap const &x, Map &z) const
{
  z = x;
  waves_->forward(z);
  CMap zm(z.data(), z.size());
  thresh_.apply(α, zm, z);
  waves_->adjoint(z);
}

} // namespace rl
