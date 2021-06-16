#pragma once

#include "fft_plan.h"
#include "tensorOps.h"

namespace FFT {

template <int TRank, int FRank>
Plan<TRank, FRank>::Plan(Tensor &workspace, Log &log, long const nThreads)
    : dims_{workspace.dimensions()}
    , log_{log}
    , threaded_{nThreads > 1}
{
  plan(workspace, nThreads);
}

template <int TRank, int FRank>
Plan<TRank, FRank>::Plan(TensorDims const &dims, Log &log, long const nThreads)
    : dims_{dims}
    , log_{log}
    , threaded_{nThreads > 1}
{
  Tensor ws(dims);
  plan(ws, nThreads);
}

template <int TRank, int FRank>
void Plan<TRank, FRank>::plan(Tensor &ws, long const nThreads)
{
  int sizes[FRank];
  int N = 1;

  // Process the two different kinds of dimensions - howmany / FFT
  {
    constexpr int FStart = TRank - FRank;
    int ii = 0;
    for (; ii < FStart; ii++) {
      N *= ws.dimension(ii);
    }
    for (; ii < TRank; ii++) {
      int const sz = ws.dimension(ii);
      sizes[ii - FStart] = sz;
      phase_[ii - FStart] = FFT::Phase(sz); // Prep FFT phase factors
    }
  }

  auto const Nvox = (sizes[0] * sizes[1] * sizes[2]);
  scale_ = 1. / sqrt(Nvox);
  auto ptr = reinterpret_cast<fftwf_complex *>(ws.data());
  log_.info(FMT_STRING("Planning {} {} FFTs with {} threads"), N, fmt::join(sizes, "x"), nThreads);

  // FFTW is row-major. Reverse dims as per
  // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
  std::reverse(&sizes[0], &sizes[FRank]);
  auto const start = log_.now();
  fftwf_plan_with_nthreads(nThreads);
  forward_plan_ = fftwf_plan_many_dft(
      FRank, sizes, N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_FORWARD, FFTW_MEASURE);
  reverse_plan_ = fftwf_plan_many_dft(
      FRank, sizes, N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_BACKWARD, FFTW_MEASURE);

  if (forward_plan_ == NULL) {
    log_.fail("Could not create forward FFT plan");
  }
  if (reverse_plan_ == NULL) {
    log_.fail("Could not create reverse FFT plan");
  }

  log_.debug("FFT planning took {}", log_.toNow(start));
}

template <int TRank, int FRank>
Plan<TRank, FRank>::~Plan()
{
  fftwf_destroy_plan(forward_plan_);
  fftwf_destroy_plan(reverse_plan_);
}

template <int TRank, int FRank>
float Plan<TRank, FRank>::scale() const
{
  return scale_;
}

template <int TRank, int FRank>
void Plan<TRank, FRank>::applyPhase(Tensor &x, float const scale, bool const forward) const
{
  auto start = log_.now();
  constexpr int FStart = TRank - FRank;
  for (long ii = 0; ii < FRank; ii++) {
    Eigen::array<long, TRank> rsh, brd;
    for (long in = 0; in < TRank; in++) {
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

template <int TRank, int FRank>
void Plan<TRank, FRank>::forward(Tensor &x) const
{
  for (long ii = 0; ii < TRank; ii++) {
    assert(x.dimension(ii) == dims_[ii]);
  }
  applyPhase(x, 1.f, true);
  auto const start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(forward_plan_, ptr, ptr);
  log_.debug("Forward FFT: {}", log_.toNow(start));
  applyPhase(x, scale_, true);
}

template <int TRank, int FRank>
void Plan<TRank, FRank>::reverse(Tensor &x) const
{
  for (long ii = 0; ii < TRank; ii++) {
    assert(x.dimension(ii) == dims_[ii]);
  }
  applyPhase(x, scale_, false);
  auto start = log_.now();
  auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
  fftwf_execute_dft(reverse_plan_, ptr, ptr);
  log_.debug("Reverse FFT: {}", log_.toNow(start));
  applyPhase(x, 1.f, false);
}

} // namespace FFT
