#include "thresh.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

template<typename T>
SoftThreshold<T>::SoftThreshold(float const λ)
  : Prox<T>()
  , λ_{λ}
{
  Log::Print<Log::Level::High>(FMT_STRING("Soft Threshold Prox λ {}"), λ);
}

template<typename T>
auto SoftThreshold<T>::operator()(float const α, Eigen::TensorMap<T const> x) const -> T
{
  float t = α * λ_;
  T s = (x.abs() > t).select(x * (x.abs() - t) / x.abs(), x.constant(0.f));
  Log::Print<Log::Level::High>(FMT_STRING("Soft Threshold α {} λ {} t {} |x| {} |s| {}"), α, λ_, t, Norm(x), Norm(s));
  return s;
}

template struct SoftThreshold<Cx4>;
template struct SoftThreshold<Cx5>;

} // namespace rl