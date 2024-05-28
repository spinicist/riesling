#include "grid.hpp"

#include "kernel/kernel.hpp"
#include "mapping.hpp"
#include "threads.hpp"
#include "top.hpp"

#include <mutex>

namespace rl {

GridOpts::GridOpts(args::Subparser &parser)
  : ktype(parser, "K", "Choose kernel - NN/KBn/ESn (ES3)", {'k', "kernel"}, "ES3")
  , osamp(parser, "O", "Grid oversampling factor (2)", {"osamp"}, 2.f)
  , vcc(parser, "V", "Virtual Conjugate Coils", {"vcc"})
  , batches(parser, "B", "Channel batch size (1)", {"batches"}, 1)
  , bucketSize(parser, "B", "Gridding subgrid size (32)", {"subgrid-size"}, 32)
  , splitSize(parser, "S", "Subgrid split size (16384)", {"subgrid-split"}, 16384)
{
}

namespace TOps {

template <typename Scalar, int NDim>
auto Grid<Scalar, NDim>::Make(TrajectoryN<NDim> const &traj,
                              std::string const        ktype,
                              float const              osamp,
                              Index const              nC,
                              Basis<Scalar> const     &b,
                              bool const               v,
                              Index const              bSz,
                              Index const              sSz) -> std::shared_ptr<Grid<Scalar, NDim>>
{
  return std::make_shared<Grid<Scalar, NDim>>(traj, ktype, osamp, nC, b, v, bSz, sSz);
}

template <typename Scalar, int NDim>
Grid<Scalar, NDim>::Grid(TrajectoryN<NDim> const &traj,
                         std::string const        ktype,
                         float const              osamp,
                         Index const              nC,
                         Basis<Scalar> const     &b,
                         bool const               v,
                         Index const              bSz,
                         Index const              sSz)
  : Parent(fmt::format("{}D GridOp", NDim))
  , kernel{make_kernel<Scalar, NDim>(ktype, osamp)}
  , mapping{traj, osamp, kernel->paddedWidth(), bSz, sSz}
  , basis{b}
{
  static_assert(NDim < 4);
  ishape = AddFront(mapping.cartDims, nC * (v ? 2 : 1), b.dimension(0));
  oshape = AddFront(mapping.noncartDims, nC);
  Log::Debug("Grid Dims {}", this->ishape);
  if (v) {
    Log::Print("Adding VCC");
    auto const conjTraj = TrajectoryN<NDim>(-traj.points(), traj.matrix(), traj.voxelSize());
    vccMapping = Mapping<NDim>(conjTraj, osamp, kernel->paddedWidth(), bSz, sSz);
  }
}

template <typename Scalar, int NDim>
Grid<Scalar, NDim>::Grid(
  std::shared_ptr<Kernel<Scalar, NDim>> const &k, Mapping<NDim> const m, Index const nC, Basis<Scalar> const &b, bool const v)
  : Parent(fmt::format("{}D GridOp", NDim), AddFront(m.cartDims, nC * (v ? 2 : 1), b.dimension(0)), AddFront(m.noncartDims, nC))
  , kernel{k}
  , mapping{m}
  , basis{b}
{
  static_assert(NDim < 4);
  Log::Debug("Grid Dims {}", this->ishape);
}

template <typename Scalar, int ND, bool vcc>
inline void forwardCoilDim(Sz<ND + 2>                                                   xi,
                           Sz<ND + 2>                                                   bi,
                           Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                           Eigen::Tensor<Scalar, ND + 2>                               &bGrid)
{
  for (Index ii = 0; ii < bGrid.dimensions()[0]; ii++) {
    bi[0] = ii;
    if constexpr (std::is_same<Scalar, Cx>::value && vcc) {
      xi[0] = bGrid.dimensions()[0] + ii;
      bGrid(bi) = std::conj(x(xi));
    } else {
      xi[0] = ii;
      // fmt::print(stderr, "bGrid {} bi {} x {} xi {}\n", bGrid.dimensions(), bi, x.dimensions(), xi);
      bGrid(bi) = x(xi);
    }
  }
}

template <typename Scalar, int ND, bool vcc>
inline void forwardBasisDim(Sz<ND + 2>                                                   xi,
                            Sz<ND + 2>                                                   bi,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                            Eigen::Tensor<Scalar, ND + 2>                               &bGrid)
{
  for (Index ii = 0; ii < x.dimensions()[1]; ii++) {
    xi[1] = ii;
    bi[1] = ii;
    forwardCoilDim<Scalar, ND, vcc>(xi, bi, x, bGrid);
  }
}

template <typename Scalar, int ND, bool vcc, int D>
inline void forwardSpatialDim(Sz<ND> const                                                 xSt,
                              Sz<ND + 2>                                                   xi,
                              Sz<ND + 2>                                                   sxi,
                              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                              Eigen::Tensor<Scalar, ND + 2>                               &sx)
{
  for (Index ii = 0; ii < sx.dimensions()[D + 2]; ii++) {
    xi[D + 2] = Wrap(ii + xSt[D], x.dimensions()[D + 2]);
    sxi[D + 2] = ii;
    if constexpr (D == 0) {
      forwardBasisDim<Scalar, ND, vcc>(xi, sxi, x, sx);
    } else {
      forwardSpatialDim<Scalar, ND, vcc, D - 1>(xSt, xi, sxi, x, sx);
    }
  }
}

template <typename Scalar, int NDim> void Grid<Scalar, NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto outer_task = [&x, &y, &basis = this->basis, &kernel = this->kernel, &ishape = this->ishape](Mapping<NDim> const &map,
                                                                                                   bool const           vcc) {
    Index const nC = ishape[0] / (vcc ? 2 : 1);
    Index const nB = ishape[1];
    auto        grid_task = [&](Index const ibucket) {
      auto const &subgrid = map.buckets[ibucket];

      auto const bSz = AddFront(subgrid.size(), nC, nB);
      InTensor   bGrid(bSz);

      Sz<NDim + 2> xi, bi;
      xi.fill(0);
      bi.fill(0);
      if (vcc) {
        forwardSpatialDim<Scalar, NDim, true, NDim - 1>(subgrid.minCorner, xi, bi, x, bGrid);
      } else {
        forwardSpatialDim<Scalar, NDim, false, NDim - 1>(subgrid.minCorner, xi, bi, x, bGrid);
      }

      for (auto ii = 0; ii < subgrid.count(); ii++) {
        auto const si = subgrid.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const o = map.offset[si];
        T1 const   bs = basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1));
        y.template chip<2>(n.trace).template chip<1>(n.sample) +=
          kernel->gather(c, o, subgrid.minCorner, bs, map.cartDims, bGrid);
      }
    };
    Threads::For(grid_task, map.buckets.size());
  };

  auto const time = this->startForward(x);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  outer_task(this->mapping, false);
  if (this->vccMapping) {
    outer_task(this->vccMapping.value(), true);
    y.device(Threads::GlobalDevice()) = y / y.constant(2.f);
  }
  this->finishForward(y, time);
}

template <typename Scalar, int NDim> void Grid<Scalar, NDim>::adjoint(OutCMap const &y, InMap &x) const
{
  auto outer_task = [&y, &x, &basis = this->basis, &kernel = this->kernel, &ishape = this->ishape](Mapping<NDim> const &map,
                                                                                                   bool const           conj) {
    Index const nC = ishape[0];
    Index const nB = ishape[1];

    std::mutex writeMutex;
    auto       grid_task = [&](Index ibucket) {
      auto const &subgrid = map.buckets[ibucket];
      auto const  bSz = AddFront(subgrid.size(), nC, nB);
      InTensor    bGrid(bSz);
      bGrid.setZero();
      for (auto ii = 0; ii < subgrid.count(); ii++) {
        auto const si = subgrid.indices[ii];
        auto const c = map.cart[si];
        auto const n = map.noncart[si];
        auto const o = map.offset[si];
        T1 const   bs =
          basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1)).conjugate();
        Eigen::Tensor<Scalar, 1> yy = y.template chip<2>(n.trace).template chip<1>(n.sample);
        kernel->spread(c, o, subgrid.minCorner, bs, yy, bGrid);
      }

      if (conj) { bGrid = bGrid.conjugate(); }

      {
        std::scoped_lock lock(writeMutex);
        auto const       gSt = AddFront(subgrid.minCorner, 0, 0); // Make this the same dims as bSz / ishape
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
    Threads::For(grid_task, map.buckets.size());
  };

  auto const time = this->startAdjoint(y);
  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  outer_task(this->mapping, false);
  if (this->vccMapping) {
    outer_task(this->vccMapping.value(), true);
    x.device(Threads::GlobalDevice()) = x / x.constant(2.f);
  }
  this->finishAdjoint(x, time);
}

template struct Grid<float, 1>;
template struct Grid<float, 2>;
template struct Grid<float, 3>;
template struct Grid<Cx, 1>;
template struct Grid<Cx, 2>;
template struct Grid<Cx, 3>;

} // namespace TOps
} // namespace rl