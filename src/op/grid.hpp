#pragma once

#include "cropper.h"
#include "fft/fft.hpp"
#include "gridBase.hpp"
#include "threads.h"

#include <mutex>

namespace {

inline Index Reflect(Index const ii, Index const sz)
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

template <int IP, int TP, typename Scalar = Cx>
struct Grid final : GridBase<Scalar>
{
  using typename GridBase<Scalar>::Input;
  using typename GridBase<Scalar>::Output;
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;

  SizedKernel<IP, TP> const *kernel;
  R2 basis;

  Grid(SizedKernel<IP, TP> const *k, Mapping const &mapping, Index const nC)
    : GridBase<Scalar>(mapping, nC, mapping.frames)
    , kernel{k}
  {
    Log::Debug(FMT_STRING("Grid<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  Grid(SizedKernel<IP, TP> const *k, Mapping const &mapping, Index const nC, R2 const b)
    : GridBase<Scalar>(mapping, nC, mapping.frames)
    , kernel{k}
    , basis{b}
  {
    Log::Debug(FMT_STRING("Grid<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  Output A(Input const &cart) const
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
    auto const &map = this->mapping_;
    bool const hasBasis = (basis.size() > 0);
    float const scale = map.scale * (hasBasis ? sqrt(basis.dimension(0)) : 1.f);

    auto grid_task = [&](Index const ii) {
      auto const si = map.sortedIndices[ii];
      auto const c = map.cart[si];
      auto const n = map.noncart[si];
      auto const ifr = map.frame[si];
      auto const k = this->kernel->k(map.offset[si]);

      Index const stX = c.x - ((IP - 1) / 2);
      Index const stY = c.y - ((IP - 1) / 2);
      Index const stZ = c.z - ((TP - 1) / 2);
      Index const btp = hasBasis ? n.spoke % basis.dimension(0) : 0;
      Eigen::Tensor<Scalar, 1> sum(nC);
      sum.setZero();
      for (Index iz = 0; iz < TP; iz++) {
        Index const iiz = Reflect(stZ + iz, cdims[4]);
        for (Index iy = 0; iy < IP; iy++) {
          Index const iiy = Reflect(stY + iy, cdims[3]);
          for (Index ix = 0; ix < IP; ix++) {
            Index const iix = Reflect(stX + ix, cdims[2]);
            float const kval = k(ix, iy, iz) * scale;
            if (hasBasis) {
              for (Index ib = 0; ib < nB; ib++) {
                float const bval = basis(btp, ib) * kval;
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
      noncart.chip(n.spoke, 2).chip(n.read, 1) = sum;
    };

    Threads::For(grid_task, map.cart.size(), "Grid Forward");
    return noncart;
  }

  Input &Adj(Output const &noncart) const
  {
    Log::Debug("Grid Adjoint");
    if (noncart.dimensions() != this->outputDimensions()) {
      Log::Fail(
        FMT_STRING("Noncartesian k-space dims {} did not match {}"), noncart.dimensions(), this->outputDimensions());
    }
    auto const &cdims = this->inputDimensions();
    auto const &map = this->mapping_;
    Index const nC = cdims[0];
    Index const nB = cdims[1];
    bool const hasBasis = (basis.size() > 0);
    float const scale = map.scale * (hasBasis ? sqrt(basis.dimension(0)) : 1.f);

    std::mutex writeMutex;
    auto grid_task = [&](Index ibucket) {
      auto const &bucket = map.buckets[ibucket];
      auto const bSz = bucket.gridSize();
      Input out(AddFront(bSz, nC, nB));
      out.setZero();

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const k = this->kernel->k(map.offset[si]);
        auto const ifr = map.frame[si];
        auto const frscale = scale * (this->weightFrames_ ? map.frameWeights[ifr] : 1.f);

        Index const stX = c.x - ((IP - 1) / 2) - bucket.minCorner[0];
        Index const stY = c.y - ((IP - 1) / 2) - bucket.minCorner[1];
        Index const stZ = c.z - ((TP - 1) / 2) - bucket.minCorner[2];
        Index const btp = hasBasis ? n.spoke % basis.dimension(0) : 0;
        Eigen::Tensor<Scalar, 1> const sample = noncart.chip(n.spoke, 2).chip(n.read, 1);
        for (Index iz = 0; iz < TP; iz++) {
          Index const iiz = stZ + iz;
          for (Index iy = 0; iy < IP; iy++) {
            Index const iiy = stY + iy;
            for (Index ix = 0; ix < IP; ix++) {
              Index const iix = stX + ix;
              float const kval = k(ix, iy, iz) * frscale;
              if (hasBasis) {
                for (Index ib = 0; ib < nB; ib++) {
                  float const bval = kval * basis(btp, ib);
                  for (Index ic = 0; ic < nC; ic++) {
                    out(ic, ib, iix, iiy, iiz) += noncart(ic, n.read, n.spoke) * bval;
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

      {
        std::scoped_lock lock(writeMutex);
        for (Index iz = 0; iz < bSz[2]; iz++) {
          Index const iiz = Reflect(bucket.minCorner[2] + iz, cdims[4]);
          for (Index iy = 0; iy < bSz[1]; iy++) {
            Index const iiy = Reflect(bucket.minCorner[1] + iy, cdims[3]);
            for (Index ix = 0; ix < bSz[0]; ix++) {
              Index const iix = Reflect(bucket.minCorner[0] + ix, cdims[2]);
              for (Index ifr = 0; ifr < nB; ifr++) {
                for (Index ic = 0; ic < nC; ic++) {
                  this->ws_->operator()(ic, ifr, iix, iiy, iiz) += out(ic, ifr, ix, iy, iz);
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
    return *(this->ws_);
  }

  R3 apodization(Sz3 const sz) const
  {
    auto gridSz = this->mapping().cartDims;
    Cx3 temp(gridSz);
    auto const fft = FFT::Make<3, 3>(gridSz);
    temp.setZero();
    auto const k = kernel->k(Point3{0, 0, 0});
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    Log::Tensor(temp, "apo-kernel");
    fft->reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale = sqrt(Product(gridSz));
    Log::Print(FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Tensor(a, "apo-final");
    return a;
  }
};

} // namespace rl
