#pragma once

#include "cropper.h"
#include "fft/fft.hpp"
#include "gridBase.hpp"
#include "threads.h"

#include "tensorOps.h"

#include <mutex>

namespace {

inline Index Wrap(Index const ii, Index const sz)
{
  if (ii < 0)
    return sz + ii;
  else if (ii >= sz)
    return ii - sz;
  else
    return ii;
}

} // namespace

namespace rl {

template <typename Scalar, typename Kernel>
struct Grid final : GridBase<Scalar>
{
  using typename GridBase<Scalar>::Input;
  using typename GridBase<Scalar>::Output;
  static size_t const NDim = Kernel::NDim;
  static size_t const kW = Kernel::PadWidth;

  Kernel kernel;
  Mapping<NDim> mapping;
  std::optional<Re2> basis;

  Sz5 inputDims_;
  Sz3 outputDims_;
  std::shared_ptr<Input> ws_;

  Grid(Trajectory const &traj, float const osamp, Index const nC, std::optional<Re2> const &b)
    : GridBase<Scalar>()
    , kernel{osamp}
    , mapping{traj, Kernel::PadWidth, osamp}
    , basis{b}
    , outputDims_{AddFront(mapping.noncartDims, nC)}
  {
    if constexpr (NDim == 3) {
      inputDims_ = AddFront(mapping.cartDims, nC, basis ? basis.value().dimension(0) : mapping.frames);
    } else {
      inputDims_ = AddFront(mapping.cartDims, nC, basis ? basis.value().dimension(0) : mapping.frames, 1);
    }
    ws_ = std::make_shared<Input>(inputDimensions());
    Log::Debug(FMT_STRING("Grid Dims {}"), this->inputDimensions());
  }

  Sz3 outputDimensions() const
  {
    return outputDims_;
  }

  Sz5 inputDimensions() const
  {
    return inputDims_;
  }

  std::shared_ptr<Input> workspace() const
  {
    return ws_;
  }

  Output forward(Input const &cart) const
  {
    if (cart.dimensions() != this->inputDimensions()) {
      Log::Fail(FMT_STRING("Cartesian k-space dims {} did not match {}"), cart.dimensions(), this->inputDimensions());
    }
    Output noncart(this->outputDimensions());
    Log::Debug("Zeroing grid output");
    noncart.device(Threads::GlobalDevice()) = noncart.constant(0.f);
    auto const &cdims = this->inputDimensions();
    Index const nC = cdims[0];
    Index const nB = cdims[1];
    auto const &map = this->mapping;
    float const scale = basis ? sqrt(basis.value().dimension(0)) : 1.f;

    auto grid_task = [&](Index const ibucket) {
      auto const &bucket = map.buckets[ibucket];

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const ifr = map.frame[si];
        auto const k = this->kernel.k(map.offset[si]);

        Index const kW_2 = ((kW - 1) / 2);
        Index const btp = basis ? n.trace % basis.value().dimension(0) : 0;
        Eigen::Tensor<Scalar, 1> sum(nC);
        sum.setZero();
        if constexpr (NDim == 3) {
          for (Index iz = 0; iz < kW; iz++) {
            Index const iiz = Wrap(c[2] - kW_2 + iz, cdims[4]);
            for (Index iy = 0; iy < kW; iy++) {
              Index const iiy = Wrap(c[1] - kW_2 + iy, cdims[3]);
              for (Index ix = 0; ix < kW; ix++) {
                Index const iix = Wrap(c[0] - kW_2 + ix, cdims[2]);
                float const kval = k(ix, iy, iz) * scale;
                if (basis) {
                  for (Index ib = 0; ib < nB; ib++) {
                    float const bval = basis.value()(btp, ib) * kval;
                    for (Index ic = 0; ic < nC; ic++) {
                      sum(ic) += cart(ic, ib, iix, iiy, iiz) * bval;
                    }
                  }
                } else {
                  for (Index ic = 0; ic < nC; ic++) {
                    sum(ic) += cart(ic, ifr, iix, iiy, iiz) * kval;
                  }
                }
              }
            }
          }
        }
        if (B0((sum.abs() < sum.abs().constant(std::numeric_limits<float>::infinity())).all())()) {
          noncart.chip(n.trace, 2).chip(n.sample, 1) = sum;
        } else {
          Log::Fail("Non-finite values {} at sample {} trace {}", Transpose(sum), n.sample, n.trace);
        }
      }
    };

    Threads::For(grid_task, map.buckets.size(), "Grid Forward");
    return noncart;
  }

  Input &adjoint(Output const &noncart) const
  {
    Log::Debug("Grid Adjoint");
    if (noncart.dimensions() != this->outputDimensions()) {
      Log::Fail(
        FMT_STRING("Noncartesian k-space dims {} did not match {}"), noncart.dimensions(), this->outputDimensions());
    }
    auto const &cdims = this->inputDimensions();
    auto const &map = this->mapping;
    Index const nC = cdims[0];
    Index const nB = cdims[1];
    float const scale = basis ? sqrt(basis.value().dimension(0)) : 1.f;

    std::mutex writeMutex;
    auto grid_task = [&](Index ibucket) {
      auto const &bucket = map.buckets[ibucket];
      auto const bSz = bucket.gridSize();
      Input out(inputDimensions());
      out.setZero();

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const k = this->kernel.k(map.offset[si]);
        auto const ifr = map.frame[si];
        auto const frscale = scale * (this->weightFrames_ ? map.frameWeights[ifr] : 1.f);

        if constexpr (NDim == 3) {
          Index const stX = c[0] - ((kW - 1) / 2) - bucket.minCorner[0];
          Index const stY = c[1] - ((kW - 1) / 2) - bucket.minCorner[1];
          Index const stZ = c[2] - ((kW - 1) / 2) - bucket.minCorner[2];
          Index const btp = basis ? n.trace % basis.value().dimension(0) : 0;
          Eigen::Tensor<Scalar, 1> const sample = noncart.chip(n.trace, 2).chip(n.sample, 1);
          for (Index iz = 0; iz < kW; iz++) {
            Index const iiz = stZ + iz;
            for (Index iy = 0; iy < kW; iy++) {
              Index const iiy = stY + iy;
              for (Index ix = 0; ix < kW; ix++) {
                Index const iix = stX + ix;
                float const kval = k(ix, iy, iz) * frscale;
                if (basis) {
                  for (Index ib = 0; ib < nB; ib++) {
                    float const bval = kval * basis.value()(btp, ib);
                    for (Index ic = 0; ic < nC; ic++) {
                      out(ic, ib, iix, iiy, iiz) += noncart(ic, n.sample, n.trace) * bval;
                    }
                  }
                } else {
                  for (Index ic = 0; ic < nC; ic++) {
                    out(ic, ifr, iix, iiy, iiz) += sample(ic) * kval;
                  }
                }
              }
            }
          }
        }
      }

      {
        std::scoped_lock lock(writeMutex);
        if constexpr (NDim == 3) {
          for (Index iz = 0; iz < bSz[2]; iz++) {
            Index const iiz = Wrap(bucket.minCorner[2] + iz, cdims[4]);
            for (Index iy = 0; iy < bSz[1]; iy++) {
              Index const iiy = Wrap(bucket.minCorner[1] + iy, cdims[3]);
              for (Index ix = 0; ix < bSz[0]; ix++) {
                Index const iix = Wrap(bucket.minCorner[0] + ix, cdims[2]);
                for (Index ifr = 0; ifr < nB; ifr++) {
                  for (Index ic = 0; ic < nC; ic++) {
                    this->ws_->operator()(ic, ifr, iix, iiy, iiz) += out(ic, ifr, ix, iy, iz);
                  }
                }
              }
            }
          }
        }
      }
    };

    Log::Debug("Zeroing workspace");
    this->ws_->device(Threads::GlobalDevice()) = this->ws_->constant(0.f);
    Threads::For(grid_task, map.buckets.size(), "Grid Adjoint");
    Log::Debug("Grid Adjoint finished");
    Log::Tensor(*(this->ws_), "grid-adjoint");
    return *(this->ws_);
  }

  Re3 apodization(Sz3 const sz) const
  {
    Cx3 temp(LastN<3>(inputDimensions()));
    auto const fft = FFT::Make<3, 3>(temp);
    temp.setZero();
    auto const k = kernel.k(Kernel::KPoint::Zero());
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    Log::Tensor(temp, "apo-kernel");
    fft->reverse(temp);
    Re3 a = Crop3(Re3(temp.real()), sz);
    float const scale = std::sqrt(Product(sz));
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Print(FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a.inverse();
    Log::Tensor(a, "apo-final");
    return a;
  }
};

} // namespace rl
