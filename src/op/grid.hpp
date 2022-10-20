#pragma once

#include "fft/fft.hpp"
#include "gridBase.hpp"
#include "mapping.hpp"
#include "pad.hpp"
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

template <typename Scalar_, typename Kernel>
struct Grid final : GridBase<Scalar_, Kernel::NDim>
{
  static constexpr size_t NDim = Kernel::NDim;
  static constexpr Index kW = Kernel::PadWidth;

  OP_INHERIT(Scalar_, NDim + 2, 3)

  Mapping<NDim> mapping;
  Kernel kernel;
  std::optional<Re2> basis;

  Grid(Mapping<NDim> const m, Index const nC, std::optional<Re2> const &b = std::nullopt)
    : GridBase<Scalar, NDim>(AddFront(m.cartDims, nC, b ? b.value().dimension(1) : m.frames), AddFront(m.noncartDims, nC))
    , mapping{m}
    , kernel{mapping.osamp}
    , basis{b}
  {
    static_assert(NDim < 4);
    Log::Print<Log::Level::High>(FMT_STRING("Grid Dims {}"), this->inputDimensions());
  }

  auto forward(InputMap x) const -> OutputMap
  {
    auto const time = this->startForward(x);
    this->output().device(Threads::GlobalDevice()) = this->output().constant(0.f);
    Index const nC = this->inputDimensions()[0];
    Index const nB = basis ? this->inputDimensions()[1] : 1;
    float const scale = basis ? std::sqrt(basis.value().dimension(0)) : 1.f;
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
                  this->output()(ic, n.sample, n.trace) += x(ic, ib, ii1) * bval;
                }
              }
            } else {
              for (Index ic = 0; ic < nC; ic++) {
                this->output()(ic, n.sample, n.trace) += x(ic, ifr, ii1) * kval;
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
                      this->output()(ic, n.sample, n.trace) += x(ic, ib, ii2, ii1) * bval;
                    }
                  }
                } else {
                  for (Index ic = 0; ic < nC; ic++) {
                    this->output()(ic, n.sample, n.trace) += x(ic, ifr, ii2, ii1) * kval;
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
                        this->output()(ic, n.sample, n.trace) += x(ic, ib, ii3, ii2, ii1) * bval;
                      }
                    }
                  } else {
                    for (Index ic = 0; ic < nC; ic++) {
                      this->output()(ic, n.sample, n.trace) += x(ic, ifr, ii3, ii2, ii1) * kval;
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
    this->finishForward(this->output(), time);
    return this->output();
  }

  auto adjoint(OutputMap y) const -> InputMap
  {
    auto const time = this->startAdjoint(y);
    auto const &map = this->mapping;
    Index const nC = this->inputDimensions()[0];
    Index const nB = basis ? this->inputDimensions()[1] : map.frames;
    float const scale = basis ? std::sqrt(basis.value().dimension(0)) : 1.f;
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
        Index constexpr hW = kW / 2;
        if (basis) {
          Index const btp = basis ? n.trace % basis.value().dimension(0) : 0;
          for (Index ib = 0; ib < nB; ib++) {
            float const bval = basis.value()(btp, ib);
            for (Index ic = 0; ic < nC; ic++) {
              bSample(ic, ib) = y(ic, n.sample, n.trace) * bval;
            }
          }
        } else {
          for (Index ib = 0; ib < nB; ib++) {
            if (ib == ifr) {
              for (Index ic = 0; ic < nC; ic++) {
                bSample(ic, ib) = y(ic, n.sample, n.trace) * scale;
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
                float const kval = k(i2, i1) * scale;
                for (Index ib = 0; ib < nB; ib++) {
                  for (Index ic = 0; ic < nC; ic++) {
                    bGrid(ic, ib, ii2, ii1) += bSample(ic, ib) * kval;
                  }
                }
              } else {
                for (Index i3 = 0; i3 < kW; i3++) {
                  Index const ii3 = i3 + c[NDim - 3] - hW - bucket.minCorner[NDim - 3];
                  float const kval = k(i3, i2, i1) * scale;
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
                this->input()(ic, ib, ii1) += bGrid(ic, ib, i1);
              }
            }
          } else {
            for (Index i2 = 0; i2 < bSz[NDim - 2]; i2++) {
              Index const ii2 = Wrap(bucket.minCorner[NDim - 2] + i2, cdims[NDim - 2]);
              if constexpr (NDim == 2) {
                for (Index ib = 0; ib < nB; ib++) {
                  for (Index ic = 0; ic < nC; ic++) {
                    this->input()(ic, ib, ii2, ii1) += bGrid(ic, ib, i2, i1);
                  }
                }
              } else {
                for (Index i3 = 0; i3 < bSz[NDim - 3]; i3++) {
                  Index const ii3 = Wrap(bucket.minCorner[NDim - 3] + i3, cdims[NDim - 3]);
                  for (Index ib = 0; ib < nB; ib++) {
                    for (Index ic = 0; ic < nC; ic++) {
                      this->input()(ic, ib, ii3, ii2, ii1) += bGrid(ic, ib, i3, i2, i1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    };

    this->input().device(Threads::GlobalDevice()) = this->input().constant(0.f);
    Threads::For(grid_task, map.buckets.size(), "Grid Adjoint");
    this->finishAdjoint(this->input(), time);
    return this->input();
  }

  auto apodization(Sz<NDim> const sz) const -> Eigen::Tensor<float, NDim>
  {
    Eigen::Tensor<Cx, NDim> temp(LastN<NDim>(this->inputDimensions()));
    auto const fft = FFT::Make<NDim, NDim>(temp);
    temp.setZero();
    float const scale = std::sqrt(Product(sz));
    Eigen::Tensor<Cx, NDim> k = kernel(Kernel::Point::Zero()).template cast<Cx>();
    PadOp<Cx, NDim, NDim> padK(k.dimensions(), temp.dimensions());
    temp = padK.forward(k * k.constant(scale));
    fft->reverse(temp);
    PadOp<Cx, NDim, NDim> padA(sz, temp.dimensions());
    Eigen::Tensor<float, NDim> a = padA.adjoint(temp).abs();
    a.device(Threads::GlobalDevice()) = a.inverse();
    Sz<NDim> center;
    std::transform(sz.begin(), sz.end(), center.begin(), [](Index i) { return i / 2; });
    LOG_DEBUG("Apodization size {} Scale: {} Norm: {} Val: {}", a.dimensions(), scale, Norm(a), a(center));
    return a;
  }
};

} // namespace rl
