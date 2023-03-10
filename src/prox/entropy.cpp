#include "entropy.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

template<typename Tensor>
Entropy<Tensor>::Entropy(float const λ, float const scale)
  : Prox<Tensor>()
  , λ_{λ}
  , scale_{scale}
{
  Log::Print(FMT_STRING("Entropy Prox λ {} scale {}"), λ, scale);
}

template<typename Tensor>
auto Entropy<Tensor>::operator()(float const α, Eigen::TensorMap<Tensor const> v) const -> Tensor
{
  using RealTensor = Eigen::Tensor<float, Tensor::NumDimensions>;
  float const t = α * λ_;
  RealTensor const vabs = v.abs() / scale_;
  RealTensor x = vabs;
  for (int ii = 0; ii < 8; ii++) {
      Log::Print(FMT_STRING("|x| {}"), Norm(x));
      x.device(Threads::GlobalDevice()) = (x > 0.f).select((x - (t/2.f) * (x.log() + 1.f)).cwiseMax(0.f), x.constant(0.f));
  }
  Log::Print(FMT_STRING("|x| {} |v| {} |vabs| {}"), Norm(x), Norm(v), Norm(vabs));
  Tensor const s = (vabs > 0.f).select(v * (scale_ * x / vabs).template cast<typename Tensor::Scalar>(), v.constant(0.f));
  Log::Print(FMT_STRING("Entropy α {} λ {} t {} |v| {} |s| {}"), α, λ_, t, Norm(v), Norm(s));
  return s;
}

template struct Entropy<Cx4>;
template struct Entropy<Cx5>;

NMREnt::NMREnt(float const λ, float const scale)
  : Prox<Cx4>()
  , λ_{λ}
  , scale_{scale}
{
  Log::Print(FMT_STRING("NMR Entropy Prox λ {}"), λ);
}

auto NMREnt::operator()(float const α, Eigen::TensorMap<Cx4 const> v) const -> Cx4
{
  float const t = α * λ_;
  Re4 const vabs = v.abs() / scale_;
  Re4 x = vabs;
  for (int ii = 0; ii < 8; ii++) {
      auto const &xx = (x.square() + 1.f).sqrt();
      x.device(Threads::GlobalDevice()) =
        (x - (t / 2.f) * ((x * (x / xx + 1.f)) / (x + xx) + (x + xx).log() - x / xx)).cwiseMax(0.f);
  }
  Cx4 const s = v * (scale_ * x / vabs).cast<Cx>();
  Log::Print(FMT_STRING("NMR Entropy α {} λ {} t {} |v| {} |s| {}"), α, λ_, t, Norm(v), Norm(s));
  return s;
}

} // namespace rl