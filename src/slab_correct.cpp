#include "slab_correct.h"

#include "fft1.h"
#include "threads.h"

double sinc(const double x)
{
  static double const taylor_0_bound = std::numeric_limits<double>::epsilon();
  static double const taylor_2_bound = sqrt(taylor_0_bound);
  static double const taylor_n_bound = sqrt(taylor_2_bound);

  if (std::abs(x) >= taylor_n_bound) {
    return (sin(x) / x);
  } else {
    // approximation by taylor series in x at 0 up to order 0
    double result = 1;

    if (abs(x) >= taylor_0_bound) {
      double x2 = x * x;
      // approximation by taylor series in x at 0 up to order 2
      result -= x2 / 6;

      if (abs(x) >= taylor_2_bound) {
        // approximation by taylor series in x at 0 up to order 4
        result += (x2 * x2) / 120;
      }
    }
    return result;
  }
}

void slab_correct(Info const &info, float const pw_us, float const rbw_kHz, Cx3 &ks, Log &log)
{
  log.info(
    FMT_STRING("Applying slab profile correction for pulse-width {} us, bandwidth {} kHz"),
    pw_us,
    rbw_kHz);
  FFT::Start(log);
  Index const N = 2 * info.read_points;
  float const os = (2.f * info.read_points) / info.matrix.maxCoeff();
  float const fov = (info.matrix.cast<float>() * info.voxel_size * 1e-3).maxCoeff();
  float const os_fov = os * fov;
  float const pw = pw_us * 1e-6;
  float const rbw = rbw_kHz * 1e3;
  auto const r = Eigen::ArrayXf::LinSpaced(N, -os_fov / 2.f, os_fov / 2.f);
  Eigen::ArrayXf const p = r.unaryExpr([fov, rbw, pw](float const rv) {
    if (rv == 0.f) {
      return 1.f;
    } else {
      float const x = M_PI * rbw * pw * rv / fov;
      return fabs(sin(x) / x);
    }
  });
  Eigen::TensorMap<R1 const> const profile(p.data(), N);

  FFT1DReal2Complex fft(N, log);
  float const beta = 1.e-1; // Regularize sinc nulls
  auto spoke_task = [&](Index const spoke_lo, Index const spoke_hi) {
    for (Index is = spoke_lo; is < spoke_hi; is++) {
      for (Index ic = 0; ic < info.channels; ic++) {
        auto phase = std::polar(1.f, std::arg(ks(ic, 0, is)));
        Cx1 spoke = ks.chip(is, 2).chip(ic, 0) / phase;
        R1 projection = fft.reverse(spoke);
        projection = (projection + beta) / (profile + beta);
        ks.chip(is, 2).chip(ic, 0) = fft.forward(projection) * phase;
      }
    }
  };
  //   spoke_task(0, ks.dimension(2));
  Threads::RangeFor(spoke_task, 0, ks.dimension(2));
  FFT::End(log);
}