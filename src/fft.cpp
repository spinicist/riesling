#include "fft.h"

#include "threads.h"
#include <filesystem>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

namespace {
std::filesystem::path WisdomPath()
{
  struct passwd *pw = getpwuid(getuid());
  const char *homedir = pw->pw_dir;
  return std::filesystem::path(homedir) / ".riesling-wisdom";
}
} // namespace

void FFTStart(Log &log)
{
  fftwf_init_threads();
  if (fftwf_import_wisdom_from_filename(WisdomPath().string().c_str())) {
    log.info("Read wisdom successfully");
  } else {
    log.info("Could not read wisdom");
  }
}

void FFTEnd(Log &log)
{
  auto const &wp = WisdomPath();
  if (fftwf_export_wisdom_to_filename(wp.string().c_str())) {
    log.info(FMT_STRING("Saved wisdom to {}"), wp.string());
  } else {
    log.info("Failed to save wisdom");
  }
  // Get use after free errors if this is called before fftw_plan_destroy in the
  // destructors below.
  // fftwf_cleanup_threads();
}

namespace {

void FFTShift3(Cx3 &data)
{
  const auto &sz = data.dimensions();
  Dims3 blk_sz{sz[0] / 2, sz[1] / 2, sz[2] / 2};
  if (blk_sz[2] == 0) { // Special case 1 slice
    auto shift_task = [&](long const lo, long const hi) {
      for (int iy = lo; iy < hi; iy++) {
        for (int ix = 0; ix < blk_sz[0]; ix++) {
          std::swap(data(ix, iy, 0), data(blk_sz[0] + ix, blk_sz[1] + iy, 0));
          std::swap(data(blk_sz[0] + ix, iy, 0), data(ix, blk_sz[1] + iy, 0));
        }
      }
    };
    Threads::RangeFor(shift_task, blk_sz[1]);
  } else {
    auto shift_task = [&](long const lo, long const hi) {
      for (int iz = lo; iz < hi; iz++) {
        for (int iy = 0; iy < blk_sz[1]; iy++) {
          for (int ix = 0; ix < blk_sz[0]; ix++) {
            std::swap(data(ix, iy, iz), data(blk_sz[0] + ix, blk_sz[1] + iy, blk_sz[2] + iz));
            std::swap(data(blk_sz[0] + ix, iy, iz), data(ix, blk_sz[1] + iy, blk_sz[2] + iz));
            std::swap(data(ix, blk_sz[1] + iy, iz), data(blk_sz[0] + ix, iy, blk_sz[2] + iz));
            std::swap(data(ix, iy, blk_sz[2] + iz), data(blk_sz[0] + ix, blk_sz[1] + iy, iz));
          }
        }
      }
    };
    Threads::RangeFor(shift_task, blk_sz[2]);
  }
} // namespace

} // namespace

FFT3::FFT3(Cx3 &grid, Log &log)
    : grid_{grid}
    , log_{log}
{
  int sizes[3];
  auto const &dims = grid.dimensions();
  std::copy_n(&dims[0], 3, &sizes[0]); // Avoid explicit casting
  auto const Nvox = (sizes[0] * sizes[1] * sizes[2]);
  scale_ = 1. / sqrt(Nvox);
  auto const start = log.start_time();
  auto ptr = reinterpret_cast<fftwf_complex *>(grid.data());
  fftwf_plan_with_nthreads(Threads::GlobalThreadCount());
  log_.info(FMT_STRING("Planning {} FFT..."), dims);
  forward_plan_ = fftwf_plan_many_dft(
      3, sizes, 1, ptr, nullptr, 1, 1, ptr, nullptr, 1, 1, FFTW_FORWARD, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_many_dft(
      3, sizes, 1, ptr, nullptr, 1, 1, ptr, nullptr, 1, 1, FFTW_BACKWARD, FFTW_MEASURE);
  log_.stop_time(start, "Took");
}

FFT3::~FFT3()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

void FFT3::forward() const
{
  auto const start = log_.start_time();
  FFTShift3(grid_);
  fftwf_execute(forward_plan_);
  grid_.device(Threads::GlobalDevice()) = grid_ * grid_.constant(scale_);
  log_.stop_time(start, "Forward FFT");
}

void FFT3::reverse() const
{
  auto const start = log_.start_time();
  fftwf_execute(reverse_plan_);
  FFTShift3(grid_);
  grid_.device(Threads::GlobalDevice()) = grid_ * grid_.constant(scale_);
  log_.stop_time(start, "Reverse FFT");
}

void FFT3::shift() const
{
  FFTShift3(grid_);
}

namespace {

void FFTShift3N(Cx4 &data)
{
  const auto &sz = data.dimensions();
  // Allow for odd number of "slices"
  Dims3 blk_sz{sz[1] / 2, sz[2] / 2, sz[3] / 2};
  if (blk_sz[2] == 0) { // Special case one slice
    auto shift_task = [&](long const lo, long const hi) {
      for (int iy = lo; iy < hi; iy++) {
        for (int ix = 0; ix < blk_sz[0]; ix++) {
          auto swap = [&](long const x1, long const x2, long const y1, long const y2) {
            Cx1 T(sz[0]);
            T = data.chip(0, 3).chip(y1 + iy, 2).chip(x1 + ix, 1);
            data.chip(0, 3).chip(y1 + iy, 2).chip(x1 + ix, 1) =
                data.chip(0, 3).chip(y2 + iy, 2).chip(x2 + ix, 1);
            data.chip(0, 3).chip(y2 + iy, 2).chip(x2 + ix, 1) = T;
          };
          swap(0, blk_sz[0], 0, blk_sz[1]);
          swap(blk_sz[0], 0, 0, blk_sz[1]);
        }
      }
    };
    Threads::RangeFor(shift_task, blk_sz[1]);
  } else {
    auto shift_task = [&](long const loz, long const hiz) {
      for (int iz = loz; iz < hiz; iz++) {
        for (int iy = 0; iy < blk_sz[1]; iy++) {
          for (int ix = 0; ix < blk_sz[0]; ix++) {
            auto swap = [&](long const x1,
                            long const x2,
                            long const y1,
                            long const y2,
                            long const z1,
                            long const z2) {
              Cx1 T(sz[0]);
              T = data.chip(z1 + iz, 3).chip(y1 + iy, 2).chip(x1 + ix, 1);
              data.chip(z1 + iz, 3).chip(y1 + iy, 2).chip(x1 + ix, 1) =
                  data.chip(z2 + iz, 3).chip(y2 + iy, 2).chip(x2 + ix, 1);
              data.chip(z2 + iz, 3).chip(y2 + iy, 2).chip(x2 + ix, 1) = T;
            };
            swap(0, blk_sz[0], 0, blk_sz[1], 0, blk_sz[2]);
            swap(blk_sz[0], 0, 0, blk_sz[1], 0, blk_sz[2]);
            swap(0, blk_sz[0], blk_sz[1], 0, 0, blk_sz[2]);
            swap(0, blk_sz[0], 0, blk_sz[1], blk_sz[2], 0);
          }
        }
      }
    };
    Threads::RangeFor(shift_task, blk_sz[2]);
  }
} // namespace

} // namespace

FFT3N::FFT3N(Cx4 &grid, Log &log)
    : grid_{grid}
    , log_{log}
{
  auto const &dims = grid.dimensions();
  int const N = dims[0];
  std::vector<int> sizes(3);
  std::copy_n(&dims[1], 3, sizes.begin()); // Avoid explicit casting
  auto const Nvox = (sizes[0] * sizes[1] * sizes[2]);
  scale_ = 1. / sqrt(Nvox);
  auto ptr = reinterpret_cast<fftwf_complex *>(grid.data());
  fftwf_plan_with_nthreads(Threads::GlobalThreadCount());
  log_.info(FMT_STRING("Planning {} FFT"), dims);
  auto const start = log_.start_time();
  forward_plan_ = fftwf_plan_many_dft(
      3, sizes.data(), N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_FORWARD, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_many_dft(
      3, sizes.data(), N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_BACKWARD, FFTW_MEASURE);
  log_.stop_time(start, "Finished planning FFTs");
}

FFT3N::~FFT3N()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

void FFT3N::forward() const
{
  auto const start = log_.start_time();
  FFTShift3N(grid_);
  fftwf_execute(forward_plan_);
  grid_.device(Threads::GlobalDevice()) = grid_ * grid_.constant(scale_);
  log_.stop_time(start, "Forward FFT");
}

void FFT3N::reverse() const
{
  auto const start = log_.start_time();
  fftwf_execute(reverse_plan_);
  FFTShift3N(grid_);
  grid_.device(Threads::GlobalDevice()) = grid_ * grid_.constant(scale_);
  log_.stop_time(start, "Reverse FFT");
}

void FFT3N::shift() const
{
  FFTShift3N(grid_);
}

namespace {

void FFTShift1(Eigen::ArrayXf &real)
{
  Eigen::Index hsz = real.size() / 2;
  Eigen::ArrayXf temp = real.head(hsz);
  real.head(hsz) = real.tail(hsz);
  real.tail(hsz) = temp;
}

} // namespace

FFT1DReal2Complex::FFT1DReal2Complex(long const N, Log &log)
    : N_{N}
    , log_{log}
{
  Eigen::ArrayXf real(N_);
  Eigen::ArrayXcf complex(N_ / 2 + 1);
  scale_ = 1. / sqrt(N_);
  auto const start = log.start_time();
  fftwf_plan_with_nthreads(Threads::GlobalThreadCount());
  log_.info("Planning FFTs...");
  auto cptr = reinterpret_cast<fftwf_complex *>(complex.data());
  forward_plan_ = fftwf_plan_dft_r2c_1d(real.size(), real.data(), cptr, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_dft_c2r_1d(real.size(), cptr, real.data(), FFTW_MEASURE);
  log_.stop_time(start, "Took");
}

FFT1DReal2Complex::~FFT1DReal2Complex()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

Eigen::ArrayXcf FFT1DReal2Complex::forward(Eigen::ArrayXf const &in) const
{
  assert(in.rows() == N_);
  Eigen::ArrayXf real = in * scale_; // This FFT is destructive
  Eigen::ArrayXcf complex(N_ / 2 + 1);
  FFTShift1(real);
  auto cptr = reinterpret_cast<fftwf_complex *>(complex.data());
  fftwf_execute_dft_r2c(forward_plan_, real.data(), cptr);
  return complex.head(N_ / 2);
}

Eigen::ArrayXf FFT1DReal2Complex::reverse(Eigen::ArrayXcf const &in) const
{
  assert(in.rows() == N_ / 2);
  Eigen::ArrayXf real(N_);
  Eigen::ArrayXcf complex(N_ / 2 + 1);
  auto cptr = reinterpret_cast<fftwf_complex *>(complex.data());
  complex.head(N_ / 2) = in * scale_;
  complex.tail(1) = 0.f;
  fftwf_execute_dft_c2r(reverse_plan_, cptr, real.data());
  FFTShift1(real);
  return real;
}
