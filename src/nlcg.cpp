#include "fft.h"
#include "nlcg.h"
#include "threads.h"

// Non-linear Conjugate Gradients with l1-regularisation
// See http://doi.wiley.com/10.1002/mrm.21391
Cx3 nlcg(
    Cx3 &radial,                     // Raw radial data
    Cx3::Dimensions const &dims,     // Output dimensions (not oversampled)
    EncodeFunction const &encode_ks, // FT + Radial sampling
    DecodeFunction const &decode_ks, // Gridding + FT
    long const &max_its,             // Maximum iterations
    float const &thresh,             // Stop if fractional change in cost-function is below thresh
    float const &lambda,             // l1-norm weight
    Log &log)
{
  float const alpha = 0.05f; // Line-search threshold (allow slightly higher residual)
  float const beta = 0.6f;   // Line-search step size will be reduced by this each iteration
  float const mu = 1.e-15f;  // Smoothing parameter for l1-norm calculation

  // Create our reference image and scale it so the regularisation and residual balance
  Cx3 x0(dims);
  decode_ks(radial, x0);
  R0 scale = x0.abs().maximum();
  x0.device(Threads::GlobalDevice()) = (1.f / scale()) * x0;
  radial.device(Threads::GlobalDevice()) = (1.f / scale()) * radial;

  // Calculate cost function f and gradient g at the same time
  auto f_g = [&](Cx3 const &x, Cx3 &grad) -> float {
    // Working space
    static Cx3 tr(radial.dimensions());
    static Cx3 resid(dims);

    // Calculate residual term
    encode_ks(x, tr);
    tr.device(Threads::GlobalDevice()) -= radial;
    decode_ks(tr, resid);

    // The l1-term is |x| which is not differentiable at x=0, so add a very small value to smooth
    // near x=0. The gradient of the l1-term would be +/-1 for real data, but we have complex so it
    // points in same direction as x
    R3 const W = (mu + (x.conjugate() * x).real()).sqrt();

    // Calculate the gradient
    grad.device(Threads::GlobalDevice()) =
        2.0f * resid + (lambda * x / W.cast<std::complex<float>>());

    // Calculate cost
    R0 const rsum = (resid * resid.conjugate()).real().sum();
    R0 const Wsum = W.sum();
    log.info(FMT_STRING("Residual term {} l1-Term {}"), rsum(), Wsum());
    return rsum() + lambda * Wsum();
  };

  // Starting point
  Cx3 x(dims);
  x.setZero();
  Cx3 temp_x(dims);
  Cx3 dx(dims);
  Cx3 g(dims);
  float f_current = f_g(x, g);
  float const f_start = f_current;

  dx.device(Threads::GlobalDevice()) = -g;
  R0 g_norm_old = (g * g.conjugate()).real().sum();
  float t = 1.0f;
  log.info("Starting Non-linear Conjugate Gradients");
  log.info(FMT_STRING("Scale %f Starting cost {}"), scale(), f_current);
  for (long ii = 0; ii < max_its; ii++) {
    R0 const dx_g = -(dx * g.conjugate()).real().sum(); // - because dx ~= -g

    // Line search
    float f_t;
    float line_thresh = f_current + alpha * t * dx_g();
    log.info(
        FMT_STRING("Start line search, current cost {}% threshold {}%"),
        100.f * f_current / f_start,
        100.f * line_thresh / f_start);
    for (long il = 0; il < 10; il++) {
      temp_x.device(Threads::GlobalDevice()) = x + t * dx;
      f_t = f_g(temp_x, g);
      log.info(FMT_STRING("t={} Cost {}%"), t, 100.f * f_t / f_start);
      if (f_t < line_thresh) {
        break;
      } else {
        t *= beta;
      }
    }
    x = temp_x;

    R0 const g_norm_new = (g * g.conjugate()).real().sum();
    float const gamma = g_norm_new() / g_norm_old();
    float const df = (f_current - f_t) / f_start;
    log.info(
        FMT_STRING("Iteration {} Cost change {}% Threshold {}%"), ii, df * 100.f, 100.f * thresh);

    if (fabs(df) < thresh) {
      log.info("Cost change reached threshold, stopping");
      break;
    }

    // Updates for next iteration
    dx.device(Threads::GlobalDevice()) = gamma * dx - g;
    g_norm_old = g_norm_new;
    f_current = f_t;
  }
  x = scale() * x;
  return x;
}
