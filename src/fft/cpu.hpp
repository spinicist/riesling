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
    plan(ws, nThreads);
  }

  CPU(Tensor &ws, Index const nThreads)
    : dims_(ws.dimensions())
    , threaded_{nThreads > 1}
  {
    plan(ws, nThreads);
  }

  void plan(Tensor &ws, Index const nThreads)
  {
    std::array<int, FRank> sz;
    N_ = 1;
    nVox_ = 1;
    // Process the two different kinds of dimensions - howmany / FFT
    {
      constexpr int FStart = TRank - FRank;
      int ii = 0;
      for (; ii < FStart; ii++) {
        N_ *= ws.dimension(ii);
      }
      std::array<Cx1, FRank> phases;

      for (; ii < TRank; ii++) {
        sz[ii - FStart] = ws.dimension(ii);
        nVox_ *= sz[ii - FStart];
        phases[ii - FStart] = Phase(sz[ii - FStart]); // Prep FFT phase factors
      }
      scale_ = 1. / sqrt(nVox_);
      Eigen::Tensor<Cx, FRank> tempPhase_(LastN<FRank>(dims_));
      tempPhase_.device(Threads::GlobalDevice()) = startPhase(phases);
      phase_.resize(Sz1{nVox_});
      phase_.device(Threads::GlobalDevice()) = tempPhase_.reshape(Sz1{nVox_});
    }

    auto ptr = reinterpret_cast<fftwf_complex *>(ws.data());
    Log::Print(FMT_STRING("Planning {} {} FFTs with {} threads"), N_, fmt::join(sz, "x"), nThreads);

    // FFTW is row-major. Reverse dims as per
    // http://www.fftw.org/fftw3_doc/Column_002dmajor-Format.html#Column_002dmajor-Format
    std::reverse(sz.begin(), sz.end());
    auto const start = Log::Now();
    fftwf_plan_with_nthreads(nThreads);
    forward_plan_ =
      fftwf_plan_many_dft(FRank, sz.data(), N_, ptr, nullptr, N_, 1, ptr, nullptr, N_, 1, FFTW_FORWARD, FFTW_MEASURE);
    reverse_plan_ =
      fftwf_plan_many_dft(FRank, sz.data(), N_, ptr, nullptr, N_, 1, ptr, nullptr, N_, 1, FFTW_BACKWARD, FFTW_MEASURE);

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
    Log::Debug(FMT_STRING("Reverse FFT"));
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
    Sz2 rshP{1, nVox_}, brdP{N_, 1}, rshX{N_, nVox_};
    auto const rbPhase = phase_.reshape(rshP).broadcast(brdP);
    auto xr = x.reshape(rshX);
    if (threaded_) {
      if (fwd) {
        xr.device(Threads::GlobalDevice()) = xr * rbPhase.constant(scale) * rbPhase;
      } else {
        xr.device(Threads::GlobalDevice()) = xr * rbPhase.constant(scale) / rbPhase;
      }
    } else {
      if (fwd) {
        xr = xr * rbPhase.constant(scale) * rbPhase;
      } else {
        xr = xr * rbPhase.constant(scale) / rbPhase;
      }
    }
    Log::Debug(FMT_STRING("FFT phase correction: {}"), Log::ToNow(start));
  }

  TensorDims dims_;
  Cx1 phase_;
  fftwf_plan forward_plan_, reverse_plan_;
  float scale_;
  Index N_, nVox_;
  bool threaded_;
};

} // namespace FFT
