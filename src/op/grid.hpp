#pragma once

#include "cropper.h"
#include "fft/fft.hpp"
#include "gridBase.hpp"
#include "mapping.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

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
struct Grid final : GridBase<Scalar, Kernel::NDim>
{
  static constexpr size_t NDim = Kernel::NDim;
  static constexpr Index kW = Kernel::PadWidth;
  using typename GridBase<Scalar, NDim>::Input;
  using InputDims = typename Input::Dimensions;
  using typename GridBase<Scalar, NDim>::Output;

  Kernel kernel;
  Mapping<NDim> mapping;
  std::optional<Re2> basis;

  InputDims inputDims_;
  Sz3 outputDims_;
  std::shared_ptr<Input> ws_;

  Grid(Trajectory const &traj, float const osamp, Index const nC, std::optional<Re2> const &b = std::nullopt)
    : GridBase<Scalar, NDim>()
    , kernel{osamp}
    , mapping{traj, Kernel::PadWidth, osamp}
    , basis{b}
    , inputDims_{AddFront(mapping.cartDims, nC, basis ? basis.value().dimension(1) : mapping.frames)}
    , outputDims_{AddFront(mapping.noncartDims, nC)}
  {
    static_assert(NDim < 4);
    Log::Print<Log::Level::High>(FMT_STRING("Grid Dims {}"), this->inputDimensions());
    ws_ = std::make_shared<Input>(inputDimensions());
  }

  Sz3 outputDimensions() const
  {
    return outputDims_;
  }

  InputDims inputDimensions() const
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
    noncart.device(Threads::GlobalDevice()) = noncart.constant(0.f);
    Index const nC = this->inputDimensions()[0];
    Index const nB = basis ? this->inputDimensions()[1] : 1;
    float const scale = basis ? sqrt(basis.value().dimension(0)) : 1.f;
    auto const &map = this->mapping;
    auto const &cdims = map.cartDims;

    auto grid_task = [&](Index const ibucket) {
      auto const &bucket = map.buckets[ibucket];

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const ifr = basis ? 0 : map.frame[si];
        auto const k = this->kernel(map.offset[si]);

        Index const kW_2 = ((kW - 1) / 2);
        Index const btp = basis ? n.trace % basis.value().dimension(0) : 0;
        Eigen::Tensor<Scalar, 1> sum(nC);
        sum.setZero();

        for (Index i1 = 0; i1 < kW; i1++) {
          Index const ii1 = Wrap(c[NDim - 1] - kW_2 + i1, cdims[NDim - 1]);
          if constexpr (NDim == 1) {
            float const kval = k(i1) * scale;
            if (basis) {
              for (Index ib = 0; ib < nB; ib++) {
                float const bval = kval * (basis ? basis.value()(btp, ib) : 1.f);
                for (Index ic = 0; ic < nC; ic++) {
                  noncart(ic, n.sample, n.trace) += cart(ic, ib, ii1) * bval;
                }
              }
            } else {
              for (Index ic = 0; ic < nC; ic++) {
                noncart(ic, n.sample, n.trace) += cart(ic, ifr, ii1) * kval;
              }
            }
          } else {
            for (Index i2 = 0; i2 < kW; i2++) {
              Index const ii2 = Wrap(c[NDim - 2] - kW_2 + i2, cdims[NDim - 2]);
              if constexpr (NDim == 2) {
                float const kval = k(i2, i1) * scale;
                if (basis) {
                  for (Index ib = 0; ib < nB; ib++) {
                    float const bval = kval * (basis ? basis.value()(btp, ib) : 1.f);
                    for (Index ic = 0; ic < nC; ic++) {
                      noncart(ic, n.sample, n.trace) += cart(ic, ib, ii2, ii1) * bval;
                    }
                  }
                } else {
                  for (Index ic = 0; ic < nC; ic++) {
                    noncart(ic, n.sample, n.trace) += cart(ic, ifr, ii2, ii1) * kval;
                  }
                }
              } else {
                for (Index i3 = 0; i3 < kW; i3++) {
                  Index const ii3 = Wrap(c[NDim - 3] - kW_2 + i3, cdims[NDim - 3]);
                  float const kval = k(i3, i2, i1) * scale;
                  if (basis) {
                    for (Index ib = 0; ib < nB; ib++) {
                      float const bval = kval * (basis ? basis.value()(btp, ib) : 1.f);
                      for (Index ic = 0; ic < nC; ic++) {
                        noncart(ic, n.sample, n.trace) += cart(ic, ib, ii3, ii2, ii1) * bval;
                      }
                    }
                  } else {
                    for (Index ic = 0; ic < nC; ic++) {
                      noncart(ic, n.sample, n.trace) += cart(ic, ifr, ii3, ii2, ii1) * kval;
                    }
                  }
                }
              }
            }
          }
        }
      }

      // if (!B0((noncart(ic, n.sample, n.trace).abs() < noncart(ic, n.sample,
      // n.trace).abs().constant(std::numeric_limits<float>::infinity())).all())()) {
      //   Log::Fail("Non-finite values at sample {} trace {}", n.sample, n.trace);
      // }
    };

    Threads::For(grid_task, map.buckets.size(), "Grid Forward");
    return noncart;
  }

  Input &adjoint(Output const &noncart) const
  {
    if (noncart.dimensions() != this->outputDimensions()) {
      Log::Fail(
        FMT_STRING("Noncartesian k-space dims {} did not match {}"), noncart.dimensions(), this->outputDimensions());
    }
    auto const &map = this->mapping;
    Index const nC = this->inputDimensions()[0];
    Index const nB = basis ? this->inputDimensions()[1] : map.frames;
    float const scale = basis ? sqrt(basis.value().dimension(0)) : 1.f;
    auto const &cdims = map.cartDims;

    std::mutex writeMutex;
    auto grid_task = [&](Index ibucket) {
      auto const &bucket = map.buckets[ibucket];
      auto const bSz = bucket.gridSize();
      Eigen::Tensor<Scalar, 2> bSample(nC, nB);
      Input bGrid(AddFront(bSz, nC, nB));
      bGrid.setZero();

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const k = this->kernel(map.offset[si]);
        auto const ifr = basis ? 0 : map.frame[si];
        auto const frscale = scale * (this->weightFrames_ ? map.frameWeights[ifr] : 1.f);

        Index constexpr hW = kW / 2;

        if (basis) {
          Index const btp = basis ? n.trace % basis.value().dimension(0) : 0;
          for (Index ib = 0; ib < nB; ib++) {
            float const bval = basis.value()(btp, ib);
            for (Index ic = 0; ic < nC; ic++) {
              bSample(ic, ib) = noncart(ic, n.sample, n.trace) * bval;
            }
          }
        } else {
          for (Index ib = 0; ib < nB; ib++) {
            if (ib == ifr) {
              for (Index ic = 0; ic < nC; ic++) {
                bSample(ic, ib) = noncart(ic, n.sample, n.trace) * frscale;
              }
            } else {
              for (Index ic = 0; ic < nC; ic++) {
                bSample(ic, ib) = Scalar(0.f);
              }
            }
          }
        }

        for (Index i1 = 0; i1 < kW; i1++) {
          Index const ii1 = i1 + c[NDim - 1] - hW - bucket.minCorner[NDim - 1];
          if constexpr (NDim == 1) {
            float const kval = k(i1);
            for (Index ib = 0; ib < nB; ib++) {
              for (Index ic = 0; ic < nC; ic++) {
                bGrid(ic, ib, ii1) += bSample(ic, ib) * kval;
              }
            }
          } else {
            for (Index i2 = 0; i2 < kW; i2++) {
              Index const ii2 = i2 + c[NDim - 2] - hW - bucket.minCorner[NDim - 2];
              if constexpr (NDim == 2) {
                float const kval = k(i2, i1) * frscale;
                for (Index ib = 0; ib < nB; ib++) {
                  for (Index ic = 0; ic < nC; ic++) {
                    bGrid(ic, ib, ii2, ii1) += bSample(ic, ib) * kval;
                  }
                }
              } else {
                for (Index i3 = 0; i3 < kW; i3++) {
                  Index const ii3 = i3 + c[NDim - 3] - hW - bucket.minCorner[NDim - 3];
                  float const kval = k(i3, i2, i1) * frscale;
                  for (Index ib = 0; ib < nB; ib++) {
                    for (Index ic = 0; ic < nC; ic++) {
                      bGrid(ic, ib, ii3, ii2, ii1) += bSample(ic, ib) * kval;
                    }
                  }
                }
              }
            }
          }
        }
      }

      {
        std::scoped_lock lock(writeMutex);
        for (Index i1 = 0; i1 < bSz[NDim - 1]; i1++) {
          Index const ii1 = Wrap(bucket.minCorner[NDim - 1] + i1, cdims[NDim - 1]);
          if constexpr (NDim == 1) {
            for (Index ib = 0; ib < nB; ib++) {
              for (Index ic = 0; ic < nC; ic++) {
                ws_->operator()(ic, ib, ii1) += bGrid(ic, ib, i1);
              }
            }
          } else {
            for (Index i2 = 0; i2 < bSz[NDim - 2]; i2++) {
              Index const ii2 = Wrap(bucket.minCorner[NDim - 2] + i2, cdims[NDim - 2]);
              if constexpr (NDim == 2) {
                for (Index ib = 0; ib < nB; ib++) {
                  for (Index ic = 0; ic < nC; ic++) {
                    ws_->operator()(ic, ib, ii2, ii1) += bGrid(ic, ib, i2, i1);
                  }
                }
              } else {
                for (Index i3 = 0; i3 < bSz[NDim - 3]; i3++) {
                  Index const ii3 = Wrap(bucket.minCorner[NDim - 3] + i3, cdims[NDim - 3]);
                  for (Index ib = 0; ib < nB; ib++) {
                    for (Index ic = 0; ic < nC; ic++) {
                      ws_->operator()(ic, ib, ii3, ii2, ii1) += bGrid(ic, ib, i3, i2, i1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    };

    this->ws_->device(Threads::GlobalDevice()) = this->ws_->constant(0.f);
    Threads::For(grid_task, map.buckets.size(), "Grid Adjoint");
    return *(this->ws_);
  }

  Re3 apodization(Sz3 const sz) const
  {
    Cx3 temp(LastN<3>(inputDimensions()));
    auto const fft = FFT::Make<3, 3>(temp);
    temp.setZero();
    auto const k = kernel(Kernel::Point::Zero());
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    fft->reverse(temp);
    Re3 a = Crop3(Re3(temp.real()), sz);
    float const scale = std::sqrt(Product(sz));
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Print(FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a.inverse();
    return a;
  }
};

} // namespace rl
