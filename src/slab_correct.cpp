#include "slab_correct.h"

#include "fft.h"
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
  FFTStart(log);
  long const N = 2 * info.read_points;
  float const os = (2.f * info.read_points) / info.matrix.maxCoeff();
  float const fov = (info.matrix.cast<float>() * info.voxel_size * 1e-3).maxCoeff();
  float const os_fov = os * fov;
  float const pw = pw_us * 1e-6;
  float const rbw = rbw_kHz * 1e3;
  auto const r = Eigen::ArrayXf::LinSpaced(N, -os_fov / 2.f, os_fov / 2.f);
  auto const profile = r.unaryExpr([fov, rbw, pw](float const rv) {
    if (rv == 0.f) {
      return 1.f;
    } else {
      float const x = M_PI * rbw * pw * rv / fov;
      return fabs(sin(x) / x);
    }
  });

  FFT1DReal2Complex fft(N, log);
  float const beta = 1.e-1; // Regularize sinc nulls
  auto spoke_task = [&](long const spoke_lo, long const spoke_hi) {
    for (long is = spoke_lo; is < spoke_hi; is++) {
      for (long ic = 0; ic < info.channels; ic++) {
        auto phase = std::polar(1.f, std::arg(ks(ic, 0, is)));
        Eigen::ArrayXcf spoke(N / 2);
        for (long ir = 0; ir < info.read_points; ir++) {
          spoke(ir) = ks(ic, ir, is) / phase;
        }
        Eigen::ArrayXf projection = fft.reverse(spoke);
        projection = (projection + beta) / (profile + beta);
        Eigen::ArrayXcf corrected = fft.forward(projection);
        for (long ir = 0; ir < info.read_points; ir++) {
          ks(ic, ir, is) = corrected(ir) * phase;
        }
      }
    }
  };
  //   spoke_task(0, ks.dimension(2));
  Threads::RangeFor(spoke_task, 0, ks.dimension(2));
  FFTEnd(log);
}