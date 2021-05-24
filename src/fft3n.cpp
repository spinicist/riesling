#include "fft3n.h"

#include "tensorOps.h"

FFT3N::FFT3N(Cx4 &grid, Log &log, long const nThreads)
    : grid_{grid}
    , log_{log}
    , threaded_{nThreads > 1}
{
  auto const &dims = grid.dimensions();
  int const N = dims[0];
  int sizes[3];
  // FFTW is row-major. Reverse dims as per
  // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
  sizes[0] = static_cast<int>(dims[3]);
  sizes[1] = static_cast<int>(dims[2]);
  sizes[2] = static_cast<int>(dims[1]);

  auto const Nvox = (sizes[0] * sizes[1] * sizes[2]);
  scale_ = 1. / sqrt(Nvox);
  auto ptr = reinterpret_cast<fftwf_complex *>(grid.data());
  log_.info(FMT_STRING("Planning {} FFT with {} threads"), dims, nThreads);
  auto const start = log_.now();
  fftwf_plan_with_nthreads(nThreads);
  forward_plan_ = fftwf_plan_many_dft(
      3, sizes, N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_FORWARD, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_many_dft(
      3, sizes, N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_BACKWARD, FFTW_MEASURE);

  if (forward_plan_ == NULL) {
    log.fail("Could not create forward FFT3N plan");
  }
  if (reverse_plan_ == NULL) {
    log.fail("Could not create reverse FFT3N plan");
  }

  phX_ = FFT::Phase(dims[1]);
  phY_ = FFT::Phase(dims[2]);
  phZ_ = FFT::Phase(dims[3]);

  log_.debug("FFT planning took {}", log_.toNow(start));
}

FFT3N::~FFT3N()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

float FFT3N::scale() const
{
  return scale_;
}

void FFT3N::forwardPhase(Cx4 &x, float const scale) const
{

  auto start = log_.now();
  if (threaded_) {
    x.device(Threads::GlobalDevice()) =
        x * x.constant(scale) *
        TileToMatch(Outer(Outer(phX_, phY_), phZ_).cast<Cx>(), x.dimensions());
  } else {
    x = x * x.constant(scale) *
        TileToMatch(Outer(Outer(phX_, phY_), phZ_).cast<Cx>(), x.dimensions());
  }
  log_.debug("Forward PhaseN: {}", log_.toNow(start));
}

void FFT3N::reversePhase(Cx4 &x, float const scale) const
{

  auto start = log_.now();
  if (threaded_) {
    auto dev = Threads::GlobalDevice();
    x.device(dev) = x * x.constant(scale) /
                    TileToMatch(Outer(Outer(phX_, phY_), phZ_).cast<Cx>(), x.dimensions());
  } else {
    x = x * x.constant(scale) /
        TileToMatch(Outer(Outer(phX_, phY_), phZ_).cast<Cx>(), x.dimensions());
  }
  log_.debug("Reverse PhaseN: {}", log_.toNow(start));
}

void FFT3N::forward() const
{
  forward(grid_);
}

void FFT3N::forward(Cx4 &x) const
{
  assert(x.dimension(1) == phX_.size());
  assert(x.dimension(2) == phY_.size());
  assert(x.dimension(3) == phZ_.size());
  forwardPhase(x, 1.f);
  auto const start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(forward_plan_, ptr, ptr);
  log_.debug("Forward FFTN: {}", log_.toNow(start));
  forwardPhase(x, scale_);
}

void FFT3N::reverse() const
{
  reverse(grid_);
}

void FFT3N::reverse(Cx4 &x) const
{
  assert(x.dimension(1) == phX_.size());
  assert(x.dimension(2) == phY_.size());
  assert(x.dimension(3) == phZ_.size());
  reversePhase(x, scale_);
  auto start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(reverse_plan_, ptr, ptr);
  log_.debug("Reverse FFTN: {}", log_.toNow(start));
  reversePhase(x, 1.f);
}
