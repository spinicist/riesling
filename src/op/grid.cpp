#include "grid.hpp"

#include "kernel/kernel.hpp"
#include "mapping.hpp"
#include "tensorop.hpp"
#include "threads.hpp"

#include <mutex>

namespace rl {

template <typename Scalar, int NDim>
auto Grid<Scalar, NDim>::Make(Trajectory const    &traj,
                              std::string const    ktype,
                              float const          osamp,
                              Index const          nC,
                              Basis<Scalar> const &b,
                              Index const          bSz,
                              Index const          sSz) -> std::shared_ptr<Grid<Scalar, NDim>>
{
  auto kernel = make_kernel<Scalar, NDim>(ktype, osamp);
  auto mapping = Mapping<NDim>(traj, osamp, kernel->paddedWidth(), bSz, sSz);
  return std::make_shared<Grid<Scalar, NDim>>(kernel, mapping, nC, b);
}

template <typename Scalar, int NDim>
Grid<Scalar, NDim>::Grid(std::shared_ptr<Kernel<Scalar, NDim>> const &k,
                         Mapping<NDim> const                          m,
                         Index const                                  nC,
                         Basis<Scalar> const                         &b)
  : Parent(fmt::format("{}D GridOp", NDim), AddFront(m.cartDims, nC, b.dimension(0)), AddFront(m.noncartDims, nC))
  , kernel{k}
  , mapping{m}
  , basis{b}
{
  static_assert(NDim < 4);
  Log::Print<Log::Level::High>("Grid Dims {}", this->ishape);
}

template <typename Scalar, int NDim>
void Grid<Scalar, NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  Index const nC = this->ishape[0];
  Index const nB = this->ishape[1];
  auto const &map = this->mapping;

  auto grid_task = [&](Index const ibucket) {
    auto const &bucket = map.buckets[ibucket];

    auto const bSz = AddFront(bucket.bucketSize(), nC, nB);
    InTensor   bGrid(bSz);

    auto const gSt = AddFront(bucket.minCorner, 0, 0); // Make this the same dims as bSz / ishape
    for (Index i1 = 0; i1 < bSz[InRank - 1]; i1++) {
      Index const w1 = Wrap(i1 + gSt[InRank - 1], ishape[InRank - 1]);
      for (Index i2 = 0; i2 < bSz[InRank - 2]; i2++) {
        Index const w2 = Wrap(i2 + gSt[InRank - 2], ishape[InRank - 2]);
        for (Index i3 = 0; i3 < bSz[InRank - 3]; i3++) {
          Index const w3 = Wrap(i3 + gSt[InRank - 3], ishape[InRank - 3]);
          if constexpr (NDim == 1) {
            bGrid(i3, i2, i1) = x(i3, i2, w1);
          } else {
            for (Index i4 = 0; i4 < bSz[InRank - 4]; i4++) {
              if constexpr (NDim == 2) {
                bGrid(i4, i3, i2, i1) = x(i4, i3, w2, w1);
              } else {
                for (Index i5 = 0; i5 < bSz[InRank - 5]; i5++) {
                  bGrid(i5, i4, i3, i2, i1) = x(i5, i4, w3, w2, w1);
                }
              }
            }
          }
        }
      }
    }

    for (auto ii = 0; ii < bucket.size(); ii++) {
      auto const si = bucket.indices[ii];
      auto const c = map.cart[si];
      auto const n = map.noncart[si];
      auto const o = map.offset[si];
      T1 const   bs = basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1));
      y.template chip<2>(n.trace).template chip<1>(n.sample) =
        this->kernel->gather(c, o, bucket.minCorner, bs, map.cartDims, bGrid);
    }
  };

  Threads::For(grid_task, map.buckets.size());
  this->finishForward(y, time);
}

template <typename Scalar, int NDim>
void Grid<Scalar, NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const  time = this->startAdjoint(y);
  auto const &map = this->mapping;
  Index const nC = this->ishape[0];
  Index const nB = this->ishape[1];

  std::mutex writeMutex;
  auto       grid_task = [&](Index ibucket) {
    auto const &bucket = map.buckets[ibucket];
    auto const  bSz = AddFront(bucket.bucketSize(), nC, nB);
    InTensor    bGrid(bSz);
    bGrid.setZero();
    for (auto ii = 0; ii < bucket.size(); ii++) {
      auto const si = bucket.indices[ii];
      auto const c = map.cart[si];
      auto const n = map.noncart[si];
      auto const o = map.offset[si];
      T1 const   bs =
        basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1)).conjugate();
      Eigen::Tensor<Scalar, 1> yy = y.template chip<2>(n.trace).template chip<1>(n.sample);
      this->kernel->spread(c, o, bucket.minCorner, bs, yy, bGrid);
    }

    {
      std::scoped_lock lock(writeMutex);
      auto const       gSt = AddFront(bucket.minCorner, 0, 0); // Make this the same dims as bSz / ishape
      for (Index i1 = 0; i1 < bSz[InRank - 1]; i1++) {
        Index const w1 = Wrap(i1 + gSt[InRank - 1], ishape[InRank - 1]);
        for (Index i2 = 0; i2 < bSz[InRank - 2]; i2++) {
          Index const w2 = Wrap(i2 + gSt[InRank - 2], ishape[InRank - 2]);
          for (Index i3 = 0; i3 < bSz[InRank - 3]; i3++) {
            Index const w3 = Wrap(i3 + gSt[InRank - 3], ishape[InRank - 3]);
            if constexpr (NDim == 1) {
              x(i3, i2, w1) += bGrid(i3, i2, i1);
            } else {
              for (Index i4 = 0; i4 < bSz[InRank - 4]; i4++) {
                if constexpr (NDim == 2) {
                  x(i4, i3, w2, w1) += bGrid(i4, i3, i2, i1);
                } else {
                  for (Index i5 = 0; i5 < bSz[InRank - 5]; i5++) {
                    x(i5, i4, w3, w2, w1) += bGrid(i5, i4, i3, i2, i1);
                  }
                }
              }
            }
          }
        }
      }
    }
  };

  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  Threads::For(grid_task, map.buckets.size());
  this->finishAdjoint(x, time);
}

template struct Grid<float, 1>;
template struct Grid<float, 2>;
template struct Grid<float, 3>;
template struct Grid<Cx, 1>;
template struct Grid<Cx, 2>;
template struct Grid<Cx, 3>;
} // namespace rl
