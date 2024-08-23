#include "grid.hpp"

#include "threads.hpp"
#include "top.hpp"

#include "chunkfor.hpp"
#include "grid-subgrid.hpp"

#include <mutex>
#include <numbers>

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
  TrajectoryN<NDim> const &traj, std::string const ktype, float const osamp, Index const nC, Basis::CPtr b, Index const sgW)
  -> std::shared_ptr<Grid<NDim, VCC>>
{
  return std::make_shared<Grid<NDim, VCC>>(traj, ktype, osamp, nC, b, sgW);
}

template <bool VCC, int ND> auto AddVCC(Sz<ND> const cart, Index const nC, Index const nB) -> Sz<ND + 2 + VCC>
{
  if constexpr (VCC) {
    return AddFront(cart, nB, nC, 2);
  } else {
    return AddFront(cart, nB, nC);
  }
}

template <int NDim, bool VCC>
Grid<NDim, VCC>::Grid(
  TrajectoryN<NDim> const &traj, std::string const ktype, float const osamp, Index const nC, Basis::CPtr b, Index const sgW)
  : Parent(fmt::format("{}D GridOp{}", NDim, VCC ? " VCC" : ""))
  , kernel{KernelBase<Scalar, NDim>::Make(ktype, osamp)}
  , subgridW{sgW + 2 * (kernel->paddedWidth() / 2)}
  , basis{b}
{
  static_assert(NDim < 4);

  auto const m = CalcMapping(traj, osamp, kernel->paddedWidth(), sgW);
  mappings = m.mappings;
  ishape = AddVCC<VCC>(m.cartDims, nC, basis ? basis->nB() : 1);
  oshape = AddFront(m.noncartDims, nC);
  if constexpr (VCC) {
    Log::Print("Adding VCC");
    auto const conjTraj = TrajectoryN<NDim>(-traj.points(), traj.matrix(), traj.voxelSize());
    vccMapping = CalcMapping<NDim>(conjTraj, osamp, kernel->paddedWidth(), sgW).mappings;
  }
  Log::Debug("Grid Dims {}", this->ishape);
}

/* Needs to be a functor to avoid template errors */
template <int ND, bool hasVCC, bool isVCC> struct forwardTask
{
  void operator()(std::span<Mapping<ND> const> const &mappings,
                  Index const                         subgridW,
                  Basis::CPtr const                  &basis,
                  KernelBase<Cx, ND>::Ptr const      &kernel,
                  CxNCMap<ND + 2 + hasVCC> const     &x,
                  CxNMap<3>                          &y) const
  {
    Index const nC = y.dimension(0);
    Index const nB = basis ? basis->nB() : 1;
    CxN<ND + 2> sx(AddFront(Constant<ND>(subgridW), nB, nC));
    Sz<ND>      current = mappings.front().subgrid;
    GridToSubgrid<ND, hasVCC, isVCC>(current, x, sx);
    for (auto const &m : mappings) {
      if (current != m.subgrid) {
        current = m.subgrid;
        GridToSubgrid<ND, hasVCC, isVCC>(current, x, sx);
      }
      Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, m.sample, m.trace), Sz1{nC});
      if (basis) {;
        kernel->gather(m.cart, m.offset, basis->entry(m.sample, m.trace), sx, yy);
      } else {
        kernel->gather(m.cart, m.offset, sx, yy);
      }
    }
  }
};

template <int NDim, bool VCC> void Grid<NDim, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  Threads::ChunkFor(forwardTask<NDim, VCC, false>(), this->mappings, subgridW, this->basis, this->kernel, x, y);
  if constexpr (VCC == true) {
    Threads::ChunkFor(forwardTask<NDim, VCC, true>(), this->vccMapping.value(), subgridW, this->basis, this->kernel, x, y);
  }
  this->finishForward(y, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::ChunkFor(forwardTask<NDim, VCC, false>(), this->mappings, subgridW, this->basis, this->kernel, x, y);
  if constexpr (VCC == true) {
    Threads::ChunkFor(forwardTask<NDim, VCC, true>(), this->vccMapping.value(), subgridW, this->basis, this->kernel, x, y);
  }
  this->finishForward(y, time, true);
}

template <int ND, bool hasVCC, bool isVCC> struct adjointTask
{
  void operator()(std::span<Mapping<ND> const> const &mappings,
                  std::mutex                         &writeMutex,
                  Index const                         subgridW,
                  Basis::CPtr const                  &basis,
                  KernelBase<Cx, ND>::Ptr const      &kernel,
                  CxNCMap<3> const                   &y,
                  CxNMap<ND + 2 + hasVCC>            &x) const
  {
    Index const nC = y.dimensions()[0];
    Index const nB = basis ? basis->nB() : 1;
    CxN<ND + 2> sx(AddFront(Constant<ND>(subgridW), nB, nC));
    sx.setZero();
    Sz<ND> current = mappings.front().subgrid;
    for (auto const &m : mappings) {
      if (current != m.subgrid) {
        std::scoped_lock lock(writeMutex);
        SubgridToGrid<ND, hasVCC, isVCC>(current, sx, x);
        sx.setZero();
        current = m.subgrid;
      }
      Eigen::Tensor<Cx, 1> yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      if (basis) {
        kernel->spread(m.cart, m.offset, basis->entryConj(m.sample, m.trace), yy, sx);
      } else {
        kernel->spread(m.cart, m.offset, yy, sx);
      }
    }

    {
      std::scoped_lock lock(writeMutex);
      SubgridToGrid<ND, hasVCC, isVCC>(current, sx, x);
    }
  }
};

template <int NDim, bool VCC> void Grid<NDim, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  std::mutex writeMutex;
  Threads::ChunkFor(adjointTask<NDim, VCC, false>(), this->mappings, writeMutex, subgridW, this->basis, this->kernel, y, x);
  if constexpr (VCC == true) {
    Threads::ChunkFor(adjointTask<NDim, VCC, true>(), this->vccMapping.value(), writeMutex, subgridW, this->basis, this->kernel,
                      y, x);
  }
  this->finishAdjoint(x, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  std::mutex writeMutex;
  Threads::ChunkFor(adjointTask<NDim, VCC, false>(), this->mappings, writeMutex, subgridW, this->basis, this->kernel, y, x);
  if constexpr (VCC == true) {
    Threads::ChunkFor(adjointTask<NDim, VCC, true>(), this->vccMapping.value(), writeMutex, subgridW, this->basis,
                      this->kernel, y, x);
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