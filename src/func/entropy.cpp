#include "entropy.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

// #define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/special_functions/lambert_w.hpp>

namespace rl {

Entropy::Entropy(float const λ)
  : Prox<Cx4>()
  , λ_{λ}
{
  Log::Print<Log::Level::High>(FMT_STRING("MaxEnt Prox λ {}"), λ);
}

auto Entropy::operator()(float const α, Eigen::TensorMap<Cx4 const> x) const -> Cx4
{
  double const t = α * λ_;
  Re4 const xabs = x.abs();
  Re4 const px = xabs.unaryExpr([t](float z) { return t * boost::math::lambert_w0(exp(static_cast<double>(z) / t - 1.) / t); })
                   .cast<float>();
  Cx4 const s = x * (px / xabs).cast<Cx>();
  Log::Print<Log::Level::High>(FMT_STRING("Prox Entropy α {} λ {} t {} |x| {} |s| {}"), α, λ_, t, Norm(x), Norm(s));
  return s;
}

NMREnt::NMREnt(float const λ)
  : Prox<Cx4>()
  , λ_{λ}
{
  Log::Print<Log::Level::High>(FMT_STRING("NMR Entropy Prox λ {}"), λ);
}

float grad(float const z)
{
  return -z / sqrt(pow(z, 2) + 1) + z * (z / sqrt(pow(z, 2) + 1) + 1) / (z + sqrt(pow(z, 2) + 1)) +
         log(z + sqrt(pow(z, 2) + 1));
}

auto NMREnt::operator()(float const α, Eigen::TensorMap<Cx4 const> x) const -> Cx4
{
  float const t = α * λ_;

  auto iter_prox = [t](float const v) {
    float x = v;
    float l = 0.f, u = std::numeric_limits<float>::infinity();

    for (int ii = 0; ii < 8; ii++) {
      float const h = grad(x);
      float const g = h + (x - v) / t;
      if (g > 0.f) {
        l = x - t * g;
        u = x;
      } else {
        l = x;
        u = x - t * g;
      }
      x = (l + u) / 2.f;
    }
    return x;
  };

  Re4 const xabs = x.abs();
  Re4 const px = xabs.unaryExpr(iter_prox);
  Cx4 const s = x * (px / xabs).cast<Cx>();
  Log::Print<Log::Level::High>(FMT_STRING("NMR Entropy α {} λ {} t {} |x| {} |s| {}"), α, λ_, t, Norm(x), Norm(s));

  return s;
}

} // namespace rl