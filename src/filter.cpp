#include "filter.h"

#include "fft3.h"
#include "tensorOps.h"
#include "threads.h"

// fw is Flat Width, ea is End Amplitude
Eigen::ArrayXf
RadialTukey(Eigen::Index const n, long const start_n, long const end_n, float const ea)
{
  auto const r = ArrayXl::LinSpaced(n, 0, n);
  Eigen::ArrayXf tukey = Eigen::ArrayXf::Zero(n);
  for (Eigen::Index i = 0; i < n; i++) {
    if (r[i] < start_n) {
      tukey[i] = 1.0f;
    } else if (r[i] < end_n) {
      tukey[i] = 0.5f * ((1 + ea) + (1 - ea) * cos((M_PI * (r[i] - start_n)) / (end_n - start_n)));
    } else {
      tukey[i] = 0.f;
    }
  }
  return tukey;
}

struct start_end_t
{
  float start;
  float end;
};

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

void ImageFilter(std::function<float(float const &)> const &f, Cx3 &image, Log &log)
{
  auto const sz = image.dimension(2);
  auto const sy = image.dimension(1);
  auto const sx = image.dimension(0);

  auto const hz = sz / 2;
  auto const hy = sy / 2;
  auto const hx = sx / 2;

  FFT3 fft(image, log);
  fft.forward(image);
  for (long iz = 0; iz < sz; iz++) {
    for (long iy = 0; iy < sy; iy++) {
      for (long ix = 0; ix < sx; ix++) {
        float const z = static_cast<float>(((iz + hz) % sz) - hz) / hz;
        float const y = static_cast<float>(((iy + hy) % sy) - hy) / hy;
        float const x = static_cast<float>(((ix + hz) % sz) - hx) / hx;
        float const r = sqrt(x * x + y * y + z * z);
        image(ix, iy, iz) *= f(r);
      }
    }
  }
  fft.reverse(image);
}

void ImageTukey(float const &s, float const &e, float const &h, Cx3 &x, Log &log)
{
  log.info(FMT_STRING("Applying Tukey filter width {}-{} height {}"), s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  ImageFilter(f, x, log);
}

void KSFilter(std::function<float(float const &)> const &f, Cx4 &ks, Log &log)
{
  auto const sz = ks.dimension(3);
  auto const sy = ks.dimension(2);
  auto const sx = ks.dimension(1);

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
          ks.chip(iz, 3).chip(iy, 2).chip(ix, 1) *=
              ks.chip(iz, 3).chip(iy, 2).chip(ix, 1).constant(val);
        }
      }
    }
  };
  Threads::RangeFor(task, sz);
}

void KSTukey(float const &s, float const &e, float const &h, Cx4 &x, Log &log)
{
  log.info(FMT_STRING("Applying Tukey filter width {}-{} height {}"), s, e, h);
  auto const &f = [&](float const &r) { return Tukey(r, s, e, h); };
  KSFilter(f, x, log);
}