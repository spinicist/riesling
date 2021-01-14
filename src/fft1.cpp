#include "fft1.h"

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
