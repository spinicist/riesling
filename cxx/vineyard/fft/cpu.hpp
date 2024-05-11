#pragma once

#include "fft.hpp"

#include "../log.hpp"
#include "tensors.hpp"

#include "ducc0/fft/fftnd_impl.h"

namespace rl {
namespace FFT {

template <int TRank, int FRank>
struct CPU final : FFT<TRank, FRank>
{
  using Tensor = typename FFT<TRank, FRank>::Tensor;
  using TensorDims = typename FFT<TRank, FRank>::TensorDims;
  using TensorMap = typename FFT<TRank, FRank>::TensorMap;
  /*! Will allocate a workspace during planning
   */
  CPU(TensorDims const &dims, Index const nThreads)
    : dims_{dims}
    , nThreads_{nThreads}
  {
    plan();
  }

  CPU(TensorMap ws, Index const nThreads)
    : dims_(ws.dimensions())
    , nThreads_{nThreads}
  {
    plan();
  }

  void plan()
  {
    std::array<int, FRank> sz;
    N_ = 1;
    nVox_ = 1;

    shape_.resize(TRank);
    for (Index ii = 0; ii < TRank; ii++) {
      shape_[ii] = dims_[TRank - 1 - ii];
    }
    fftAxes_.resize(FRank);
    for (Index ii = 0; ii < FRank; ii++) {
      fftAxes_[ii] = ii;
    }

    // Process the two different kinds of dimensions - howmany / FFT
    {
      constexpr int FStart = TRank - FRank;
      int           ii = 0;
      for (; ii < FStart; ii++) {
        N_ *= dims_[ii];
      }
      std::array<Cx1, FRank> phases;

      for (; ii < TRank; ii++) {
        sz[ii - FStart] = dims_[ii];
        nVox_ *= sz[ii - FStart];
        phases[ii - FStart] = Phase(sz[ii - FStart]); // Prep FFT phase factors
      }
      scale_ = 1. / sqrt(nVox_);
      Eigen::Tensor<Cx, FRank> tempPhase_(LastN<FRank>(dims_));
      tempPhase_.device(Threads::GlobalDevice()) = startPhase(phases);
      phase_.resize(Sz1{nVox_});
      phase_.device(Threads::GlobalDevice()) = tempPhase_.reshape(Sz1{nVox_});
    }
    // fmt::print(stderr, "shape {} axes {} scale_ {}\n", shape_, fftAxes_, scale_);
  }

  ~CPU() {}

  void forward(TensorMap x) const //!< Image space to k-space
  {
    for (Index ii = 0; ii < TRank; ii++) {
      assert(x.dimension(ii) == dims_[ii]);
    }
    applyPhase(x, true);
    ducc0::c2c(ducc0::cfmav(x.data(), shape_), ducc0::vfmav(x.data(), shape_), fftAxes_, true, scale_, nThreads_);
    applyPhase(x, true);
  }

  void reverse(TensorMap x) const //!< K-space to image space
  {
    for (Index ii = 0; ii < TRank; ii++) {
      assert(x.dimension(ii) == dims_[ii]);
    }
    applyPhase(x, false);
    ducc0::c2c(ducc0::cfmav(x.data(), shape_), ducc0::vfmav(x.data(), shape_), fftAxes_, false, scale_, nThreads_);
    applyPhase(x, false);
  }

private:
  TensorDims dims_;
  Cx1        phase_;
  float      scale_;
  Index      N_, nVox_, nThreads_;
  std::vector<size_t> shape_, fftAxes_;

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

  void applyPhase(TensorMap x, bool const fwd) const
  {
    Sz2        rshP{1, nVox_}, brdP{N_, 1}, rshX{N_, nVox_};
    auto const rbPhase = phase_.reshape(rshP).broadcast(brdP);
    auto       xr = x.reshape(rshX);
    if (nThreads_ > 1) {
      if (fwd) {
        xr.device(Threads::GlobalDevice()) = xr * rbPhase;
      } else {
        xr.device(Threads::GlobalDevice()) = xr / rbPhase;
      }
    } else {
      if (fwd) {
        xr = xr * rbPhase;
      } else {
        xr = xr / rbPhase;
      }
    }
  }
};

} // namespace FFT
} // namespace rl
