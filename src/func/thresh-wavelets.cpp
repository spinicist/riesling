#include "thresh-wavelets.hpp"

namespace rl {

ThresholdWavelets::ThresholdWavelets(Sz4 const dims, float const λ, Index const W, Index const L)
  : Prox<Cx4>()
  , waves_{dims, W, L}
  , λ_{λ}
{
}

auto ThresholdWavelets::operator()(float const α, Eigen::TensorMap<Cx4 const>x) const -> Cx4
{
  Cx4 temp = x;
  waves_.forward(temp);
  temp = thresh_(λ_ * α, temp);
  waves_.adjoint(temp);
  return temp;
}

} // namespace rl
