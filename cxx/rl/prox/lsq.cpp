#include "lsq.hpp"

#include "../algo/common.hpp"
#include "../log.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

template <typename S>
LeastSquares<S>::LeastSquares(float const λ_, Index const sz_)
  : Prox<S>(sz_)
  , λ{λ_}
  , y{nullptr, sz_}
{
  Log::Print("Prox", "LeastSquares Prox λ {}", λ);
}

template <typename S>
LeastSquares<S>::LeastSquares(float const λ_, CMap const bias)
  : Prox<S>(bias.rows())
  , λ{λ_}
  , y{bias}
{
  Log::Print("Prox", "LeastSquares Prox λ {}", λ);
}

template <typename S> void LeastSquares<S>::apply(float const α, CMap const &x, Map &z) const
{
  float const t = α * λ;
  if (y.data()) {
    z = (x - t * y) / (1.f + t);
  } else {
    z = x / (1.f + t);
  }
  if (Log::IsDebugging()) {
    Log::Debug("Prox", "LeastSquares α {} λ {} t {} |x| {} |y| {} |z| {}", α, λ, t, ParallelNorm(x), ParallelNorm(y),
               ParallelNorm(z));
  }
}

template <typename S> void LeastSquares<S>::apply(std::shared_ptr<Ops::Op<S>> const α, CMap const &x, Map &z) const
{
  auto const div = α->inverse(1.f, λ);
  if (y.data()) {
    z = div->forward(x - λ * α->forward(y));
  } else {
    z = div->forward(x);
  }
  if (Log::IsDebugging()) {
    Log::Debug("Prox", "LeastSquares λ {} |x| {} |y| {} |z| {}", λ, ParallelNorm(x), ParallelNorm(y), ParallelNorm(z));
  }
}

template <typename S> void LeastSquares<S>::setBias(S const *data) { new (&this->y) CMap(data, this->sz); }

template struct LeastSquares<float>;
template struct LeastSquares<Cx>;

} // namespace rl::Proxs