#include "thresh-wavelets.hpp"

namespace rl {

ThresholdWavelets::ThresholdWavelets(Sz4 const dims, Index const W, Index const L)
  : Prox<Cx4>()
  , pad_{dims, LastN<3>(Wavelets::PaddedDimensions(dims, L))}
  , waves_{pad_.outputDimensions(), W, L}
{
}

auto ThresholdWavelets::operator()(float const λ, Cx4 const &x) const -> Cx4
{
  return pad_.adjoint(waves_.adjoint(thresh_(λ, waves_.forward(pad_.forward(x)))));
}

} // namespace rl
