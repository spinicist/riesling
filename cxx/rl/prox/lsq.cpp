#include "lsq.hpp"

#include "../algo/common.hpp"
#include "../log/log.hpp"
#include "../tensors.hpp"

#include <new>

namespace rl::Proxs {

template <typename S> LeastSquares<S>::LeastSquares(float const λ_, Index const sz_)
  : Prox<S>(sz_)
  , λ{λ_}
  , y{nullptr, 0}
{
  Log::Print("Prox", "LeastSquares Prox λ {}", λ);
}

template <typename S> LeastSquares<S>::LeastSquares(float const λ_, CMap y_)
  : Prox<S>(y_.rows())
  , λ{λ_}
  , y{y_}
{
  Log::Print("Prox", "LeastSquares Prox λ {}", λ);
}

template <typename S> void LeastSquares<S>::apply(float const α, CMap x, Map z) const
{
  float const t = α * λ;
  if (y.size()) {
    z.device(Threads::TensorDevice()) = (x - t * y) / (1.f + t);
  } else {
    z.device(Threads::TensorDevice()) = x / (1.f + t);
  }
}

template <typename S> void LeastSquares<S>::apply(std::shared_ptr<Ops::Op<S>> const α, CMap x, Map z) const
{
  auto const div = α->inverse(1.f, λ);
  if (y.size()) {
    z.device(Threads::TensorDevice()) = div->forward(x - λ * α->forward(y));
  } else {
    z.device(Threads::TensorDevice()) = div->forward(x);
  }
}

template <typename S> void LeastSquares<S>::setY(CMap y_)
{
  if (y_.rows() == this->sz) {
    new (&this->y) CMap(y_.data(), y_.rows());
  } else {
    throw(Log::Failure("ProxLSQ", "New y had size {}, old is {}", y_.rows(), this->y.rows()));
  }
}

template struct LeastSquares<float>;
template struct LeastSquares<Cx>;

} // namespace rl::Proxs