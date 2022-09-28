#include "thresh-wavelets.hpp"

namespace rl {

ThresholdWavelets::ThresholdWavelets(Sz4 const dims, Index const W, Index const L, float const λ)
  : Functor<Cx4>()
  , pad_{dims, LastN<3>(Wavelets::PaddedDimensions(dims, L))}
  , waves_{pad_.outputDimensions(), W, L}
  , thresh_{λ}
{
}

auto ThresholdWavelets::operator()(Cx4 const &x) const -> Cx4
{
  return pad_.adjoint(waves_.adjoint(thresh_(waves_.forward(pad_.forward(x)))));
}

} // namespace rl
