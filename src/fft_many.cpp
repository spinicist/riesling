#include "fft_many.h"

#include "tensorOps.h"

namespace FFT {

template <int TR, int FStart, int FN>
Many<TR, FStart, FN>::Many(Tensor &workspace, Log &log, long const nThreads)
    : ws_{workspace}
    , log_{log}
    , threaded_{nThreads > 1}
{
  int sizes[FN];
  int N = 1;
  int stride = 1;

  // Process the three different kinds of dimensions - howmany / FFT / strides
  {
    int ii = 0;
    for (; ii < FStart; ii++) {
      N *= ws_.dimension(ii);
    }
    for (; ii < FStart + FN; ii++) {
      int const sz = ws_.dimension(ii);
      sizes[ii - FStart] = sz;
      phase_[ii - FStart] = FFT::Phase(sz); // Prep FFT phase factors
    }
    for (; ii < TR; ii++) {
      stride *= ws_.dimension(ii);
    }
  }

  auto const Nvox = (sizes[0] * sizes[1] * sizes[2]);
  scale_ = 1. / sqrt(Nvox);
  auto ptr = reinterpret_cast<fftwf_complex *>(ws_.data());
  log_.info(
      FMT_STRING("Planning {} {} FFTs, stride {} with {} threads"),
      N,
      fmt::join(sizes, "x"),
      stride,
      nThreads);

  // FFTW is row-major. Reverse dims as per
  // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
  std::reverse(&sizes[0], &sizes[FN]);
  auto const start = log_.now();
  fftwf_plan_with_nthreads(nThreads);
  forward_plan_ = fftwf_plan_many_dft(
      FN, sizes, N, ptr, nullptr, N, stride, ptr, nullptr, N, stride, FFTW_FORWARD, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_many_dft(
      FN, sizes, N, ptr, nullptr, N, stride, ptr, nullptr, N, stride, FFTW_BACKWARD, FFTW_MEASURE);

  if (forward_plan_ == NULL) {
    log.fail("Could not create forward FFT plan");
  }
  if (reverse_plan_ == NULL) {
    log.fail("Could not create reverse FFT plan");
  }

  log_.debug("FFT planning took {}", log_.toNow(start));
}

template <int TR, int FStart, int FN>
Many<TR, FStart, FN>::~Many()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

template <int TR, int FStart, int FN>
float Many<TR, FStart, FN>::scale() const
{
  return scale_;
}

template <int TR, int FStart, int FN>
void Many<TR, FStart, FN>::applyPhase(Tensor &x, float const scale, bool const forward) const
{

  auto start = log_.now();
  for (long ii = 0; ii < FN; ii++) {
    Eigen::array<long, TR> rsh, brd;
    for (long in = 0; in < TR; in++) {
      rsh[in] = 1;
      brd[in] = x.dimension(in);
    }
    rsh[FStart + ii] = phase_[ii].dimension(0);
    brd[FStart + ii] = 1;

    if (threaded_) {
      if (forward) {
        x.device(Threads::GlobalDevice()) = x * phase_[ii].reshape(rsh).broadcast(brd);
      } else {
        x.device(Threads::GlobalDevice()) = x / phase_[ii].reshape(rsh).broadcast(brd);
      }
    } else {
      if (forward) {
        x = x * phase_[ii].reshape(rsh).broadcast(brd);
      } else {
        x = x / phase_[ii].reshape(rsh).broadcast(brd);
      }
    }
  }
  if (scale != 1.f) {
    if (threaded_) {
      x.device(Threads::GlobalDevice()) = x * x.constant(scale);
    } else {
      x = x * x.constant(scale);
    }
  }
  log_.debug("Forward PhaseN: {}", log_.toNow(start));
} // namespace FFT

template <int TR, int FStart, int FN>
void Many<TR, FStart, FN>::forward() const
{
  forward(ws_);
}

template <int TR, int FStart, int FN>
void Many<TR, FStart, FN>::forward(Tensor &x) const
{
  for (long ii = 0; ii < TR; ii++) {
    assert(x.dimension(ii) == ws_.dimension(ii));
  }
  applyPhase(x, 1.f, true);
  auto const start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(forward_plan_, ptr, ptr);
  log_.debug("Forward FFT: {}", log_.toNow(start));
  applyPhase(x, scale_, true);
}

template <int TR, int FStart, int FN>
void Many<TR, FStart, FN>::reverse() const
{
  reverse(ws_);
}

template <int TR, int FStart, int FN>
void Many<TR, FStart, FN>::reverse(Tensor &x) const
{
  for (long ii = 0; ii < TR; ii++) {
    assert(x.dimension(ii) == ws_.dimension(ii));
  }
  applyPhase(x, scale_, false);
  auto start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(reverse_plan_, ptr, ptr);
  log_.debug("Reverse FFT: {}", log_.toNow(start));
  applyPhase(x, 1.f, false);
}

template struct Many<4, 1, 3>;

} // namespace FFT
