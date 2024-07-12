#include "grid.hpp"

#include "threads.hpp"
#include "top.hpp"

#include <mutex>
#include <numbers>

namespace {
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;
}

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

template <int NDim, bool VCC>
auto Grid<NDim, VCC>::Make(TrajectoryN<NDim> const &traj,
                           std::string const        ktype,
                           float const              osamp,
                           Index const              nC,
                           Basis const             &b,
                           Index const              bSz,
                           Index const              sSz) -> std::shared_ptr<Grid<NDim, VCC>>
{
  return std::make_shared<Grid<NDim, VCC>>(traj, ktype, osamp, nC, b, bSz, sSz);
}

template <bool VCC, int ND> auto AddVCC(Sz<ND> const cart, Index const nC, Index const nB) -> Sz<ND + 2 + VCC>
{
  if constexpr (VCC) {
    return AddFront(cart, nC, 2, nB);
  } else {
    return AddFront(cart, nC, nB);
  }
}

template <int NDim, bool VCC>
Grid<NDim, VCC>::Grid(TrajectoryN<NDim> const &traj,
                      std::string const        ktype,
                      float const              osamp,
                      Index const              nC,
                      Basis const             &b,
                      Index const              bSz,
                      Index const              sSz)
  : Parent(fmt::format("{}D GridOp{}", NDim, VCC ? " VCC" : ""))
  , kernel{Kernel<Scalar, NDim>::Make(ktype, osamp)}
  , mapping{traj, osamp, kernel->paddedWidth(), bSz, sSz}
  , basis{b}
{
  static_assert(NDim < 4);
  ishape = AddVCC<VCC>(mapping.cartDims, nC, b.dimension(0));
  oshape = AddFront(mapping.noncartDims, nC);
  if constexpr (VCC) {
    Log::Print("Adding VCC");
    auto const conjTraj = TrajectoryN<NDim>(-traj.points(), traj.matrix(), traj.voxelSize());
    vccMapping = Mapping<NDim>(conjTraj, osamp, kernel->paddedWidth(), bSz, sSz);
  }
  Log::Debug("Grid Dims {}", this->ishape);
}

template <int NDim, bool VCC>
Grid<NDim, VCC>::Grid(std::shared_ptr<Kernel<Scalar, NDim>> const &k, Mapping<NDim> const m, Index const nC, Basis const &b)
  : Parent(fmt::format("{}D GridOp{}", NDim, VCC ? " VCC" : ""),
           AddVCC<VCC>(m.cartDims, nC, b.dimension(0)),
           AddFront(m.noncartDims, nC))
  , kernel{k}
  , mapping{m}
  , basis{b}
{
  static_assert(NDim < 4);
  Log::Debug("Grid Dims {}", this->ishape);
}

template <int ND, bool hasVCC, bool isVCC>
inline void forwardTask(Mapping<ND> const                                                &map,
                        Basis const                                                      &basis,
                        std::shared_ptr<Kernel<Cx, ND>> const                            &kernel,
                        Eigen::TensorMap<Eigen::Tensor<Cx, ND + 2 + hasVCC> const> const &x,
                        Eigen::TensorMap<Eigen::Tensor<Cx, 3>>                           &y)
{
  Index const nC = y.dimension(0);
  Index const nB = basis.dimension(0);
  auto        grid_task = [&](Index const is) {
    auto const               &subgrid = map.subgrids[is];
    Eigen::Tensor<Cx, ND + 2> sx(AddFront(subgrid.size(), nC, nB));
    subgrid.template gridToSubgrid<hasVCC, isVCC>(x, sx);

    for (auto ii = 0; ii < subgrid.count(); ii++) {
      auto const                 si = subgrid.indices[ii];
      auto const                 c = map.cart[si];
      auto const                 n = map.noncart[si];
      auto const                 o = map.offset[si];
      Eigen::Tensor<Cx, 1> const bs =
        basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1));
      Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, n.sample, n.trace), Sz1{nC});
      kernel->gather(c, o, subgrid.minCorner, bs, sx, yy);
    }
  };
  Threads::For(grid_task, map.subgrids.size());
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  forwardTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, x, y);
  if (this->vccMapping) { forwardTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, x, y); }
  this->finishForward(y, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  forwardTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, x, y);
  if (this->vccMapping) { forwardTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, x, y); }
  this->finishForward(y, time, true);
}

template <int ND, bool hasVCC, bool isVCC>
inline void adjointTask(Mapping<ND> const                                    &map,
                        Basis const                                          &basis,
                        std::shared_ptr<Kernel<Cx, ND>> const                &kernel,
                        Eigen::TensorMap<Eigen::Tensor<Cx, 3> const> const   &y,
                        Eigen::TensorMap<Eigen::Tensor<Cx, ND + 2 + hasVCC>> &x)
{
  Index const nC = y.dimensions()[0];
  Index const nB = basis.dimension(0);

  std::mutex writeMutex;
  auto       grid_task = [&](Index is) {
    auto const               &subgrid = map.subgrids[is];
    Eigen::Tensor<Cx, ND + 2> sx(AddFront(subgrid.size(), nC, nB));
    sx.setZero();
    for (auto ii = 0; ii < subgrid.count(); ii++) {
      auto const                 si = subgrid.indices[ii];
      auto const                 c = map.cart[si];
      auto const                 n = map.noncart[si];
      auto const                 o = map.offset[si];
      Eigen::Tensor<Cx, 1> const bs =
        basis.template chip<2>(n.trace % basis.dimension(2)).template chip<1>(n.sample % basis.dimension(1)).conjugate();
      Eigen::Tensor<Cx, 1> yy = y.template chip<2>(n.trace).template chip<1>(n.sample);
      kernel->spread(c, o, subgrid.minCorner, bs, yy, sx);
    }

    {
      std::scoped_lock lock(writeMutex);
      subgrid.template subgridToGrid<hasVCC, isVCC>(sx, x);
    }
  };
  Threads::For(grid_task, map.subgrids.size());
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::adjoint(OutCMap const &y, InMap &x) const
{

  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  adjointTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, y, x);
  if (this->vccMapping) { adjointTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, y, x); }
  this->finishAdjoint(x, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{

  auto const time = this->startAdjoint(y, x, true);
  adjointTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, y, x);
  if (this->vccMapping) { adjointTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, y, x); }
  this->finishAdjoint(x, time, true);
}

template struct Grid<1, false>;
template struct Grid<2, false>;
template struct Grid<3, false>;
template struct Grid<1, true>;
template struct Grid<2, true>;
template struct Grid<3, true>;
} // namespace TOps
} // namespace rl