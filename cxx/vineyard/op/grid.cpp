#include "grid.hpp"

#include "threads.hpp"
#include "top.hpp"

#include "chunkfor.hpp"
#include "grid-subgrid.hpp"

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
{
}

namespace TOps {

template <int NDim, bool VCC>
auto Grid<NDim, VCC>::Make(
  TrajectoryN<NDim> const &traj, std::string const ktype, float const osamp, Index const nC, Basis const &b, Index const sgW)
  -> std::shared_ptr<Grid<NDim, VCC>>
{
  return std::make_shared<Grid<NDim, VCC>>(traj, ktype, osamp, nC, b, sgW);
}

template <bool VCC, int ND> auto AddVCC(Sz<ND> const cart, Index const nC, Index const nB) -> Sz<ND + 2 + VCC>
{
  if constexpr (VCC) {
    return AddFront(cart, nC, nB, 2);
  } else {
    return AddFront(cart, nC, nB);
  }
}

template <int NDim, bool VCC>
Grid<NDim, VCC>::Grid(
  TrajectoryN<NDim> const &traj, std::string const ktype, float const osamp, Index const nC, Basis const &b, Index const sgW)
  : Parent(fmt::format("{}D GridOp{}", NDim, VCC ? " VCC" : ""))
  , kernel{Kernel<Scalar, NDim>::Make(ktype, osamp)}
  , subgridW{sgW + 2 * (kernel->paddedWidth() / 2)}
  , basis{b}
{
  static_assert(NDim < 4);

  auto const m = CalcMapping(traj, osamp, kernel->paddedWidth(), sgW);
  mappings = m.mappings;
  ishape = AddVCC<VCC>(m.cartDims, nC, b.dimension(0));
  oshape = AddFront(m.noncartDims, nC);
  if constexpr (VCC) {
    Log::Print("Adding VCC");
    auto const conjTraj = TrajectoryN<NDim>(-traj.points(), traj.matrix(), traj.voxelSize());
    vccMapping = CalcMapping<NDim>(conjTraj, osamp, kernel->paddedWidth(), sgW).mappings;
  }
  Log::Debug("Grid Dims {}", this->ishape);
}

template <int ND, bool hasVCC, bool isVCC>
inline void forwardTask(std::ranges::viewable_range auto const &mappings,
                        Index const                             subgridW,
                        Basis const                            &basis,
                        std::shared_ptr<Kernel<Cx, ND>> const  &kernel,
                        CxNCMap<ND + 2 + hasVCC> const         &x,
                        CxNMap<3>                              &y)
{
  Index const nC = y.dimension(0);
  Index const nB = basis.dimension(0);
  CxN<ND + 2> sx(AddFront(Constant<ND>(subgridW), nC, nB));
  Sz<ND>      current = mappings.front().subgrid;
  GridToSubgrid<ND, hasVCC, isVCC>(current, x, sx);
  for (auto const &m : mappings) {
    if (current != m.subgrid) {
      current = m.subgrid;
      GridToSubgrid<ND, hasVCC, isVCC>(current, x, sx);
    }
    Eigen::Tensor<Cx, 1> const bs =
      basis.template chip<2>(m.noncart.trace % basis.dimension(2)).template chip<1>(m.noncart.sample % basis.dimension(1));
    Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, m.noncart.sample, m.noncart.trace), Sz1{nC});
    kernel->gather(m.cart, m.offset, current, bs, sx, yy);
  }
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  Threads::ChunkFor(
    [&](std::ranges::viewable_range auto const &m) {
      forwardTask<NDim, VCC, false>(m, subgridW, this->basis, this->kernel, x, y);
    },
    this->mappings);
  if constexpr (VCC == true) {
    Threads::ChunkFor(
      [&](std::ranges::viewable_range auto const &m) {
        forwardTask<NDim, VCC, true>(m, subgridW, this->basis, this->kernel, x, y);
      },
      this->vccMapping.value());
  }
  this->finishForward(y, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::ChunkFor(
    [&](std::ranges::viewable_range auto const &m) {
      forwardTask<NDim, VCC, false>(m, subgridW, this->basis, this->kernel, x, y);
    },
    this->mappings);
  if constexpr (VCC == true) {
    Threads::ChunkFor(
      [&](std::ranges::viewable_range auto const &m) {
        forwardTask<NDim, VCC, true>(m, subgridW, this->basis, this->kernel, x, y);
      },
      this->vccMapping.value());
  }
  this->finishForward(y, time, true);
}

template <int ND, bool hasVCC, bool isVCC>
inline void adjointTask(Index const                            subgridW,
                        std::vector<Mapping<ND>> const        &mappings,
                        Basis const                           &basis,
                        std::shared_ptr<Kernel<Cx, ND>> const &kernel,
                        CxNCMap<3> const                      &y,
                        CxNMap<ND + 2 + hasVCC>               &x)
{
  Index const nC = y.dimensions()[0];
  Index const nB = basis.dimension(0);

  std::mutex writeMutex;
  // auto       grid_task = [&](Index is) {

  CxN<ND + 2> sx(AddFront(Constant<ND>(subgridW), nC, nB));
  sx.setZero();
  Sz<ND> current = mappings.front().subgrid;

  for (auto const &m : mappings) {
    if (current != m.subgrid) {
      std::scoped_lock lock(writeMutex);
      SubgridToGrid<ND, hasVCC, isVCC>(current, sx, x);
      sx.setZero();
      current = m.subgrid;
    }

    Eigen::Tensor<Cx, 1> const bs = basis.template chip<2>(m.noncart.trace % basis.dimension(2))
                                      .template chip<1>(m.noncart.sample % basis.dimension(1))
                                      .conjugate();
    Eigen::Tensor<Cx, 1> yy = y.template chip<2>(m.noncart.trace).template chip<1>(m.noncart.sample);
    kernel->spread(m.cart, m.offset, current, bs, yy, sx);
  }

  {
    std::scoped_lock lock(writeMutex);
    SubgridToGrid<ND, hasVCC, isVCC>(current, sx, x);
  }
}
// Threads::For(grid_task, map.subgrids.size());

template <int NDim, bool VCC> void Grid<NDim, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  adjointTask<NDim, VCC, false>(subgridW, this->mappings, this->basis, this->kernel, y, x);
  if constexpr (VCC == true) {
    adjointTask<NDim, VCC, true>(subgridW, this->vccMapping.value(), this->basis, this->kernel, y, x);
  }
  this->finishAdjoint(x, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{

  auto const time = this->startAdjoint(y, x, true);
  adjointTask<NDim, VCC, false>(subgridW, this->mappings, this->basis, this->kernel, y, x);
  if constexpr (VCC == true) {
    adjointTask<NDim, VCC, true>(subgridW, this->vccMapping.value(), this->basis, this->kernel, y, x);
  }
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