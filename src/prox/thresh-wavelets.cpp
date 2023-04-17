#include "thresh-wavelets.hpp"

namespace rl {

ThresholdWavelets::ThresholdWavelets(float const λ, Sz4 const shape, Index const W, Index const L)
  : Prox<Cx>()
  , waves_{std::make_shared<Wavelets>(shape, W, L)}
  , thresh_{λ, shape}
{
}

void ThresholdWavelets::operator()(float const α, Vector const &x, Vector &z) const
{
  z = x;
  waves_->forward(z);
  thresh_(α, z, z);
  waves_->adjoint(z);
}

} // namespace rl
