#include "filter.h"

#include "fft_plan.h"
#include "tensorOps.h"
#include "threads.h"

inline float Tukey(float const &r, float const &sw, float const &ew, float const &eh)
{
  if (r > ew) {
    return 0.f;
  } else if (r > sw) {
    return (0.5f * ((1 + eh) + (1 - eh) * cos((M_PI * (r - sw)) / (ew - sw))));
  } else {
    return 1.f;
  }
}

template<typename Scalar, int D>
void KSFilter(std::function<float(float const &)> const &f,
              Eigen::Tensor<Scalar, D> &ks,
              Log &log)
{
  auto const sz = ks.dimension(D - 1);
  auto const sy = ks.dimension(D - 2);
  auto const sx = ks.dimension(D - 3);

  auto const hz = sz / 2;
  auto const hy = sy / 2;
  auto const hx = sx / 2;

  auto task = [&](long const lo, long const hi) {
    for (long iz = lo; iz < hi; iz++) {
      for (long iy = 0; iy < sy; iy++) {
        for (long ix = 0; ix < sx; ix++) {
          float const rz = static_cast<float>(iz - hz) / hz;
          float const ry = static_cast<float>(iy - hy) / hy;
          float const rx = static_cast<float>(ix - hx) / hx;
          float const r = sqrt(rx * rx + ry * ry + rz * rz);
          float const val = f(r);
          ks.chip(iz, D - 1).chip(iy, D - 2).chip(ix, D - 3) *= 
            ks.chip(iz, D - 1).chip(iy, D - 2).chip(ix, D - 3).constant(val);
        }
      }
    }
  };
  Threads::RangeFor(task, sz);
}

void ImageTukey(float const &s, float const &e, float const &h, Cx3 &x, Log &log)
{
  log.info(FMT_STRING("Applying Tukey filter width {}-{} height {}"), s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  FFT::ThreeD fft(x.dimensions(), log);
  log.image(x, "tukey-img-before.nii");
  fft.forward(x);
  log.image(x, "tukey-ks-before.nii");
  KSFilter(f, x, log);
  log.image(x, "tukey-ks-after.nii");
  fft.reverse(x);
  log.image(x, "tukey-img-after.nii");
}

void KSTukey(float const &s, float const &e, float const &h, Cx4 &x, Log &log)
{
  log.info(FMT_STRING("Applying Tukey filter width {}-{} height {}"), s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  KSFilter(f, x, log);
}
