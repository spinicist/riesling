#include "fft1.h"
#include "tensorOps.h"

namespace {

} // namespace

FFT1DReal2Complex::FFT1DReal2Complex(long const N, Log &log)
    : N_{N}
    , log_{log}
{
  assert(N % 2 == 0);
  R1 real(N_);
  Cx1 complex(N_ / 2 + 1);
  real.setZero();
  complex.setZero();
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

void FFT1DReal2Complex::shift(R1 &x) const
{
  assert(x.size() == N_);
  long const hsz = N_ / 2;
  R1 temp = x.slice(Sz1{0}, Sz1{hsz});
  x.slice(Sz1{0}, Sz1{hsz}) = x.slice(Sz1{hsz}, Sz1{hsz});
  x.slice(Sz1{hsz}, Sz1{hsz}) = temp;
}

void FFT1DReal2Complex::shift(Cx1 &x) const
{
  assert(x.size() == N_);
  long const hsz = N_ / 2;
  Cx1 temp = x.slice(Sz1{0}, Sz1{hsz});
  x.slice(Sz1{0}, Sz1{hsz}) = x.slice(Sz1{hsz}, Sz1{hsz});
  x.slice(Sz1{hsz}, Sz1{hsz}) = temp;
}

Cx1 FFT1DReal2Complex::forward(R1 const &in) const
{
  assert(in.size() == N_);
  R1 real = in * scale_; // This FFT is destructive
  shift(real);
  Cx1 complex(N_ / 2 + 1);
  auto cptr = reinterpret_cast<fftwf_complex *>(complex.data());
  fftwf_execute_dft_r2c(forward_plan_, real.data(), cptr);
  return complex;
}

R1 FFT1DReal2Complex::reverse(Cx1 const &in) const
{
  assert(in.size() == (N_ / 2 + 1));
  R1 real(N_);
  Cx1 temp = in; // This FFT is destructive
  auto cptr = reinterpret_cast<fftwf_complex *>(temp.data());
  fftwf_execute_dft_c2r(reverse_plan_, cptr, real.data());
  shift(real);
  real = real * real.constant(scale_);
  return real;
}
