#include "grid.hpp"

#include "kernel/kernel.hpp"
#include "mapping.hpp"
#include "threads.hpp"
#include "top.hpp"

#include <mutex>
#include <numbers>

using namespace std::numbers;

namespace rl {

GridOpts::GridOpts(args::Subparser &parser)
  : ktype(parser, "K", "Choose kernel - NN/KBn/ESn (ES3)", {'k', "kernel"}, "ES3")
  , osamp(parser, "O", "Grid oversampling factor (2)", {"osamp"}, 2.f)
  , vcc(parser, "V", "Virtual Conjugate Coils", {"vcc"})
  , batches(parser, "B", "Channel batch size (1)", {"batches"}, 1)
  , subgridSize(parser, "B", "Gridding subgrid size (32)", {"subgrid-size"}, 32)
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
                           Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                           Sz<ND + 2>                                                   sxi,
                           Eigen::Tensor<Scalar, ND + 2>                               &sx)
{
  for (Index ii = 0; ii < sx.dimensions()[0]; ii++) {
    sxi[0] = ii;
    if constexpr (std::is_same<Scalar, Cx>::value && vcc) {
      xi[0] = sx.dimensions()[0] + ii;
      sx(sxi) = std::conj(x(xi));
    } else {
      xi[0] = ii;
      sx(sxi) = x(xi);
    }
  }
}

template <typename Scalar, int ND, bool vcc>
inline void forwardBasisDim(Sz<ND + 2>                                                   xi,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                            Sz<ND + 2>                                                   sxi,
                            Eigen::Tensor<Scalar, ND + 2>                               &sx)
{
  for (Index ii = 0; ii < x.dimensions()[1]; ii++) {
    xi[1] = ii;
    sxi[1] = ii;
    forwardCoilDim<Scalar, ND, vcc>(xi, x, sxi, sx);
  }
}

template <typename Scalar, int ND, bool vcc, int D>
inline void forwardSpatialDim(Sz<ND> const                                                 xSt,
                              Sz<ND + 2>                                                   xi,
                              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                              Sz<ND + 2>                                                   sxi,
                              Eigen::Tensor<Scalar, ND + 2>                               &sx)
{
  for (Index ii = 0; ii < sx.dimensions()[D + 2]; ii++) {
    xi[D + 2] = Wrap(ii + xSt[D], x.dimensions()[D + 2]);
    sxi[D + 2] = ii;
    if constexpr (D == 0) {
      forwardBasisDim<Scalar, ND, vcc>(xi, x, sxi, sx);
    } else {
      forwardSpatialDim<Scalar, ND, vcc, D - 1>(xSt, xi, x, sxi, sx);
    }
  }
}

template <typename Scalar, int ND, bool vcc>
inline void forwardTask(Mapping<ND> const                                           &map,
                        Basis<Scalar> const                                         &basis,
                        std::shared_ptr<Kernel<Scalar, ND>> const                   &kernel,
                        Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2> const> const &x,
                        Eigen::TensorMap<Eigen::Tensor<Scalar, 3>>                  &y)
{
  Index const nC = x.dimensions()[0] / (vcc ? 2 : 1);
  Index const nB = x.dimensions()[1];
  auto        grid_task = [&](Index const is) {
    auto const                   &subgrid = map.subgrids[is];
    Eigen::Tensor<Scalar, ND + 2> sx(AddFront(subgrid.size(), nC, nB));
    Sz<ND + 2>                    xi, sxi;
    xi.fill(0);
    sxi.fill(0);
    forwardSpatialDim<Scalar, ND, vcc, ND - 1>(subgrid.minCorner, xi, x, sxi, sx);

    for (auto ii = 0; ii < subgrid.count(); ii++) {
      auto const                     si = subgrid.indices[ii];
      auto const                     c = map.cart[si];
      auto const                     n = map.noncart[si];
      auto const                     o = map.offset[si];
      Eigen::Tensor<Scalar, 1> const bs =
        basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1));
      y.template chip<2>(n.trace).template chip<1>(n.sample) += kernel->gather(c, o, subgrid.minCorner, bs, map.cartDims, sx);
    }
  };
  Threads::For(grid_task, map.subgrids.size());
}

template <typename Scalar, int NDim> void Grid<Scalar, NDim>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  forwardTask<Scalar, NDim, false>(this->mapping, this->basis, this->kernel, x, y);
  if (this->vccMapping) {
    forwardTask<Scalar, NDim, true>(this->vccMapping.value(), this->basis, this->kernel, x, y);
    y.device(Threads::GlobalDevice()) = y / y.constant(sqrt2);
  }
  this->finishForward(y, time);
}

template <typename Scalar, int ND, bool vcc>
inline void adjointCoilDim(Sz<ND + 2>                                       sxi,
                           Eigen::Tensor<Scalar, ND + 2> const             &sx,
                           Sz<ND + 2>                                       xi,
                           Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2>> &x)
{
  for (Index ii = 0; ii < sx.dimensions()[0]; ii++) {
    sxi[0] = ii;
    if constexpr (std::is_same<Scalar, Cx>::value && vcc) {
      xi[0] = sx.dimensions()[0] + ii;
      x(xi) += std::conj(sx(sxi));
    } else {
      xi[0] = ii;
      // fmt::print(stderr, "bGrid {} bi {} x {} xi {}\n", bGrid.dimensions(), bi, x.dimensions(), xi);
      x(xi) += sx(sxi);
    }
  }
}

template <typename Scalar, int ND, bool vcc>
inline void adjointBasisDim(Sz<ND + 2>                                       sxi,
                            Eigen::Tensor<Scalar, ND + 2> const             &sx,
                            Sz<ND + 2>                                       xi,
                            Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2>> &x)
{
  for (Index ii = 0; ii < x.dimensions()[1]; ii++) {
    xi[1] = ii;
    sxi[1] = ii;
    adjointCoilDim<Scalar, ND, vcc>(sxi, sx, xi, x);
  }
}

template <typename Scalar, int ND, bool vcc, int D>
inline void adjointSpatialDim(Sz<ND + 2>                                       sxi,
                              Eigen::Tensor<Scalar, ND + 2> const             &sx,
                              Sz<ND> const                                     xSt,
                              Sz<ND + 2>                                       xi,
                              Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2>> &x)
{
  for (Index ii = 0; ii < sx.dimensions()[D + 2]; ii++) {
    xi[D + 2] = Wrap(ii + xSt[D], x.dimensions()[D + 2]);
    sxi[D + 2] = ii;
    if constexpr (D == 0) {
      adjointBasisDim<Scalar, ND, vcc>(sxi, sx, xi, x);
    } else {
      adjointSpatialDim<Scalar, ND, vcc, D - 1>(sxi, sx, xSt, xi, x);
    }
  }
}

template <typename Scalar, int ND, bool vcc>
inline void adjointTask(Mapping<ND> const                                      &map,
                        Basis<Scalar> const                                    &basis,
                        std::shared_ptr<Kernel<Scalar, ND>> const              &kernel,
                        Eigen::TensorMap<Eigen::Tensor<Scalar, 3> const> const &y,
                        Eigen::TensorMap<Eigen::Tensor<Scalar, ND + 2>>        &x)
{
  Index const nC = x.dimensions()[0] / (vcc ? 2 : 1);
  Index const nB = x.dimensions()[1];

  std::mutex writeMutex;
  auto       grid_task = [&](Index is) {
    auto const                   &subgrid = map.subgrids[is];
    Eigen::Tensor<Scalar, ND + 2> sx(AddFront(subgrid.size(), nC, nB));
    sx.setZero();
    for (auto ii = 0; ii < subgrid.count(); ii++) {
      auto const                     si = subgrid.indices[ii];
      auto const                     c = map.cart[si];
      auto const                     n = map.noncart[si];
      auto const                     o = map.offset[si];
      Eigen::Tensor<Scalar, 1> const bs =
        basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1)).conjugate();
      Eigen::Tensor<Scalar, 1> yy = y.template chip<2>(n.trace).template chip<1>(n.sample);
      kernel->spread(c, o, subgrid.minCorner, bs, yy, sx);
    }

    {
      std::scoped_lock lock(writeMutex);
      Sz<ND + 2>       xi, sxi;
      xi.fill(0);
      sxi.fill(0);
      adjointSpatialDim<Scalar, ND, vcc, ND - 1>(sxi, sx, subgrid.minCorner, xi, x);
    }
  };
  Threads::For(grid_task, map.subgrids.size());
}

template <typename Scalar, int NDim> void Grid<Scalar, NDim>::adjoint(OutCMap const &y, InMap &x) const
{

  auto const time = this->startAdjoint(y);
  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  adjointTask<Scalar, NDim, false>(this->mapping, this->basis, this->kernel, y, x);
  if (this->vccMapping) {
    adjointTask<Scalar, NDim, true>(this->vccMapping.value(), this->basis, this->kernel, y, x);
    x.device(Threads::GlobalDevice()) = x / x.constant(sqrt2);
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