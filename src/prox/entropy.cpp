#include "entropy.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

template <typename Tensor>
Entropy<Tensor>::Entropy(float const λ, float const scale)
  : Prox<Tensor>()
  , λ_{λ}
  , scale_{scale}
{
  Log::Print(FMT_STRING("Entropy Prox λ {} scale {}"), λ, scale);
}

template <typename Tensor>
auto Entropy<Tensor>::operator()(float const α, Eigen::TensorMap<Tensor const> v) const -> Tensor
{
  using RealTensor = Eigen::Tensor<float, Tensor::NumDimensions>;
  float const t = α * λ_;
  RealTensor const vabs = v.abs() * scale_;
  RealTensor x = vabs;
  for (int ii = 0; ii < 16; ii++) {
    auto const g = (x > 0.f).select((x.log() + 1.f) + (1.f / t) * (x - vabs), x.constant(0.f));
    x.device(Threads::GlobalDevice()) = (x - (t / 2.f) * g).cwiseMax(0.f);
  }
  Tensor const s = (vabs > 0.f).select(v * (x / vabs).template cast<typename Tensor::Scalar>(), v.constant(0.f));
  Log::Print(FMT_STRING("Entropy α {} λ {} t {} |v| {} |s| {}"), α, λ_, t, Norm(v), Norm(s));
  return s;
}

template struct Entropy<Cx4>;
template struct Entropy<Cx5>;

NMREntropy::NMREntropy(float const λ, float const scale)
  : Prox<Cx4>()
  , λ_{λ}
  , scale_{scale}
{
  Log::Print(FMT_STRING("NMR Entropy Prox λ {} scale {}"), λ_, scale_);
}

auto NMREntropy::operator()(float const α, Eigen::TensorMap<Cx4 const> v) const -> Cx4
{
  float const t = α * λ_ * scale_;
  Re4 const vabs = v.abs() * scale_;
  Re4 x = vabs;
  for (int ii = 0; ii < 16; ii++) {
    auto const xx = (x.square() + 1.f).sqrt();
    auto const g = ((x * (x / xx + 1.f)) / (x + xx) + (x + xx).log() - x / xx) + (1.f / t) * (x - vabs);
    x.device(Threads::GlobalDevice()) = (x - (t / 2.f) * g).cwiseMax(0.f);
  }
  Cx4 const s = v * (x / vabs).cast<Cx>();
  Log::Print(FMT_STRING("NMR Entropy α {} λ {} t {} |v| {} |s| {}"), α, λ_, t, Norm(v), Norm(s));
  return s;
}

} // namespace rl