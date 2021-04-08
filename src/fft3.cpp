#include "fft3.h"

#include "tensorOps.h"

FFT3::FFT3(Sz3 const &sz, Log &log, long const nThreads)
    : log_{log}
    , threaded_{nThreads > 1}
{
  Cx3 temp(sz);
  init(temp, nThreads);
}

FFT3::FFT3(Cx3 &grid, Log &log, long const nThreads)
    : log_{log}
    , threaded_{nThreads > 1}
{
  init(grid, nThreads);
}

void FFT3::init(Cx3 &workspace, long const nThreads)
{
  auto const &dims = workspace.dimensions();
  int sizes[3];
  // FFTW is row-major. Reverse dims as per
  // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
  sizes[0] = static_cast<int>(dims[2]);
  sizes[1] = static_cast<int>(dims[1]);
  sizes[2] = static_cast<int>(dims[0]);
  auto const Nvox = (sizes[0] * sizes[1] * sizes[2]);
  scale_ = 1. / sqrt(Nvox);
  auto ptr = reinterpret_cast<fftwf_complex *>(workspace.data());
  log_.info(FMT_STRING("Planning {} FFT with {} threads"), dims, nThreads);
  auto const start = log_.now();
  fftwf_plan_with_nthreads(nThreads);
  forward_plan_ = fftwf_plan_many_dft(
      3, sizes, 1, ptr, nullptr, 1, 1, ptr, nullptr, 1, 1, FFTW_FORWARD, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_many_dft(
      3, sizes, 1, ptr, nullptr, 1, 1, ptr, nullptr, 1, 1, FFTW_BACKWARD, FFTW_MEASURE);

  phX_ = FFT::Phase(dims[0]);
  phY_ = FFT::Phase(dims[1]);
  phZ_ = FFT::Phase(dims[2]);

  log_.debug("Planning took {}", log_.toNow(start));
}

FFT3::~FFT3()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

void FFT3::forwardPhase(Cx3 &x, float const scale) const
{
  auto start = log_.now();
  if (threaded_) {
    auto dev = Threads::GlobalDevice();
    x.device(dev) = x * x.constant(scale) * Outer(Outer(phX_, phY_), phZ_).cast<Cx>();
  } else {
    x = x * x.constant(scale) * Outer(Outer(phX_, phY_), phZ_).cast<Cx>();
  }
  log_.debug("Forward FFT3 Phase: {}", log_.toNow(start));
}

void FFT3::reversePhase(Cx3 &x, float const scale) const
{
  auto start = log_.now();
  if (threaded_) {
    auto dev = Threads::GlobalDevice();
    x.device(dev) = x * x.constant(scale) / Outer(Outer(phX_, phY_), phZ_).cast<Cx>();
  } else {
    x = x * x.constant(scale) / Outer(Outer(phX_, phY_), phZ_).cast<Cx>();
  }
  log_.debug("Reverse FFT3 Phase: {}", log_.toNow(start));
}

void FFT3::forward(Cx3 &x) const
{
  assert(x.dimension(0) == phX_.size());
  assert(x.dimension(1) == phY_.size());
  assert(x.dimension(2) == phZ_.size());

  forwardPhase(x, 1.f);
  auto start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(forward_plan_, ptr, ptr);
  log_.debug("Forward FFT3: {}", log_.toNow(start));
  forwardPhase(x, scale_);
}

void FFT3::reverse(Cx3 &x) const
{
  assert(x.dimension(0) == phX_.size());
  assert(x.dimension(1) == phY_.size());
  assert(x.dimension(2) == phZ_.size());
  reversePhase(x, scale_);
  auto start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(reverse_plan_, ptr, ptr);
  log_.debug("Reverse FFT3: {}", log_.toNow(start));
  reversePhase(x, 1.f);
}
