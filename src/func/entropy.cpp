#include "entropy.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

// #define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/special_functions/lambert_w.hpp>

namespace rl {

ProxEnt::ProxEnt(float const λ)
  : Prox<Cx4>()
  , λ_{λ}
{
  Log::Print<Log::Level::High>(FMT_STRING("Soft Threshold Prox λ {}"), λ);
}

auto ProxEnt::operator()(float const α, Eigen::TensorMap<Cx4 const> x) const -> Cx4
{
  float const t = α * λ_;
  Re4 const xabs = x.abs();
  Log::Print(FMT_STRING("Check t {} |x| {} |xabs| {}"), t, Norm(x), Norm(xabs));
  Re4 const step1 = xabs / t - 1.f;
  Log::Print(FMT_STRING("Check {} {}"), Norm(step1), Maximum(step1));
  Rd4 const step2 = step1.cast<double>().exp() / (double)t;
  Log::Print(FMT_STRING("Check {} {}"), Norm(step2), Maximum(step2));
  Re4 const step3 = step2.unaryExpr([](double z) { return boost::math::lambert_w0(z); }).cast<float>();
  Log::Print(FMT_STRING("Check {} {}"), Norm(step3), Maximum(step3));
  Cx4 const s = x * (step3 * xabs.constant(t) / xabs).cast<Cx>();
  Log::Print<Log::Level::High>(FMT_STRING("Prox Entropy α {} λ {} t {} |x| {} |s| {}"), α, λ_, t, Norm(x), Norm(s));
  return s;
}

} // namespace rl