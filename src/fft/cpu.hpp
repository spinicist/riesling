#pragma once

#include "fft.hpp"

#include "../log.h"
#include "../tensorOps.h"

#include "fftw3.h"

namespace FFT {

template <int TRank, int FRank>
struct CPU final : FFT<TRank, FRank>
{
  using Tensor = typename FFT<TRank, FRank>::Tensor;
  using TensorDims = typename Tensor::Dimensions;

  /*! Will allocate a workspace during planning
   */
  CPU(TensorDims const &dims, Index const nThreads)
    : dims_{dims}
    , threaded_{nThreads > 1}
  {
    Tensor ws(dims);
    std::array<int, FRank> sizes;
    int N = 1;
    int Nvox = 1;
    // Process the two different kinds of dimensions - howmany / FFT
    {
      constexpr int FStart = TRank - FRank;
      int ii = 0;
      for (; ii < FStart; ii++) {
        N *= ws.dimension(ii);
      }
      std::array<Cx1, FRank> phases;
      for (; ii < TRank; ii++) {
        int const sz = ws.dimension(ii);
        Nvox *= sz;
        sizes[ii - FStart] = sz;
        phases[ii - FStart] = Phase(sz); // Prep FFT phase factors
      }
      scale_ = 1. / sqrt(Nvox);
      phase_.resize(LastN<FRank>(dims));
      phase_.device(Threads::GlobalDevice()) = startPhase(phases);
    }

    auto ptr = reinterpret_cast<fftwf_complex *>(ws.data());
    Log::Print(
      FMT_STRING("Planning {} {} FFTs with {} threads"), N, fmt::join(sizes, "x"), nThreads);

    // FFTW is row-major. Reverse dims as per
    // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
    std::reverse(sizes.begin(), sizes.end());
    auto const start = Log::Now();
    fftwf_plan_with_nthreads(nThreads);
    forward_plan_ = fftwf_plan_many_dft(
      FRank, sizes.data(), N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_FORWARD, FFTW_MEASURE);
    reverse_plan_ = fftwf_plan_many_dft(
      FRank, sizes.data(), N, ptr, nullptr, N, 1, ptr, nullptr, N, 1, FFTW_BACKWARD, FFTW_MEASURE);

    if (forward_plan_ == NULL) {
      Log::Fail(FMT_STRING("Could not create forward FFT Planned"));
    }
    if (reverse_plan_ == NULL) {
      Log::Fail(FMT_STRING("Could not create reverse FFT Planned"));
    }

    Log::Debug(FMT_STRING("FFT planning took {}"), Log::ToNow(start));
  }

  ~CPU()
  {
    fftwf_destroy_plan(forward_plan_);
    fftwf_destroy_plan(reverse_plan_);
  }

  void forward(Tensor &x) const //!< Image space to k-space
  {
    for (Index ii = 0; ii < TRank; ii++) {
      assert(x.dimension(ii) == dims_[ii]);
    }
    Log::Debug(FMT_STRING("Forward FFT"));
    auto const start = Log::Now();
    applyPhase(x, 1.f, true);
    auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
    fftwf_execute_dft(forward_plan_, ptr, ptr);
    applyPhase(x, scale_, true);
    Log::Debug(FMT_STRING("Forward FFT: {}"), Log::ToNow(start));
  }

  void reverse(Tensor &x) const //!< K-space to image space
  {
    for (Index ii = 0; ii < TRank; ii++) {
      assert(x.dimension(ii) == dims_[ii]);
    }
    Log::Print(FMT_STRING("Reverse FFT"));
    auto start = Log::Now();
    applyPhase(x, scale_, false);
    auto ptr = reinterpret_cast<fftwf_complex *>(x.data());
    fftwf_execute_dft(reverse_plan_, ptr, ptr);
    applyPhase(x, 1.f, false);
    Log::Debug(FMT_STRING("Reverse FFT: {}"), Log::ToNow(start));
  }

private:
  template <int D, typename T>
  decltype(auto) nextPhase(T const &x, std::array<Cx1, FRank> const &ph) const
  {
    Eigen::array<Index, FRank> rsh, brd;
    for (Index in = 0; in < FRank; in++) {
      rsh[in] = 1;
      brd[in] = ph[in].dimension(0);
    }
    if constexpr (D < FRank) {
      rsh[D] = ph[D].dimension(0);
      brd[D] = 1;
      return ph[D].reshape(rsh).broadcast(brd) * nextPhase<D + 1>(x, ph);
    } else {
      return x;
    }
  }

  decltype(auto) startPhase(std::array<Cx1, FRank> const &ph) const
  {
    Eigen::array<Index, FRank> rsh, brd;
    for (Index in = 0; in < FRank; in++) {
      rsh[in] = 1;
      brd[in] = ph[in].dimension(0);
    }
    rsh[0] = ph[0].dimension(0);
    brd[0] = 1;
    if constexpr (FRank == 1) {
      return ph[0];
    } else {
      return nextPhase<1>(ph[0].reshape(rsh).broadcast(brd), ph);
    }
  }

  void applyPhase(Tensor &x, float const scale, bool const fwd) const
  {
    auto start = Log::Now();
    TensorDims rsh, brd;
    constexpr int FStart = TRank - FRank;
    int ii = 0;
    for (; ii < FStart; ii++) {
      rsh[ii] = 1;
      brd[ii] = dims_[ii];
    }
    for (; ii < TRank; ii++) {
      rsh[ii] = dims_[ii];
      brd[ii] = 1;
    }

    auto const rbPhase = phase_.reshape(rsh).broadcast(brd);
    if (threaded_) {
      if (fwd) {
        x.device(Threads::GlobalDevice()) = x * x.constant(scale) * rbPhase;
      } else {
        x.device(Threads::GlobalDevice()) = x * x.constant(scale) / rbPhase;
      }
    } else {
      if (fwd) {
        x = x * x.constant(scale) * rbPhase;
      } else {
        x = x * x.constant(scale) / rbPhase;
      }
    }
    Log::Debug(FMT_STRING("FFT phase correction: {}"), Log::ToNow(start));
  }

  TensorDims dims_;
  Eigen::Tensor<Cx, FRank> phase_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;

  bool threaded_;
};

} // namespace FFT
