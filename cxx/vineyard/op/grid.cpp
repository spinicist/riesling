#include "grid.hpp"

#include "grid-subgrid.hpp"
#include "log.hpp"
#include "op/top-impl.hpp"
#include "sys/threads.hpp"

#include <numbers>

namespace rl {

namespace TOps {

template <int ND>
auto Grid<ND>::Make(Opts const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b) -> std::shared_ptr<Grid<ND>>
{
  return std::make_shared<Grid<ND>>(opts, traj, nC, b);
}

template <int ND>
Grid<ND>::Grid(Opts const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  : Parent(fmt::format("{}D GridOp", ND))
  , kernel{KernelBase<Scalar, ND>::Make(opts.ktype, opts.osamp)}
  , subgridW{opts.subgridSize}
  , basis{b}
{
  static_assert(ND < 4);
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  gridLists = traj.toCoordLists(osMatrix, kernel->paddedWidth(), subgridW, false);
  ishape = AddFront(osMatrix, basis ? basis->nB() : 1, nC);
  oshape = Sz3{nC, traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  if (opts.vcc) {
    Log::Print("Grid", "Adding VCC");
    ishape[1] *= 2;
    vccLists = traj.toCoordLists(osMatrix, kernel->paddedWidth(), subgridW, true);
  }
  Log::Debug("Grid", "ishape {} oshape {}", this->ishape, this->oshape);
}

/* Needs to be a functor to avoid template errors */
template <int ND>
void Grid<ND>::forwardTask(Index const                   start,
                           Index const                   stride,
                           std::vector<CoordList> const &lists,
                           bool const                    isVCC,
                           CxNCMap<ND + 2> const        &x,
                           CxNMap<3>                    &y) const
{
  Index const nC = y.dimension(0);
  Index const nB = basis ? basis->nB() : 1;
  CxN<ND + 2> sx(AddFront(Constant<ND>(SubgridFullwidth(subgridW, kernel->paddedWidth())), nB, nC));
  for (Index is = start; is < lists.size(); is += stride) {
    auto const &list = lists[is];
    GridToSubgrid<ND>(SubgridCorner(list.corner, subgridW, kernel->paddedWidth()), vccLists.has_value(), isVCC, x, sx);
    for (auto const &m : list.coords) {
      Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, m.sample, m.trace), Sz1{nC});
      if (basis) {
        kernel->gather(m.cart, m.offset, basis->entry(m.sample, m.trace), sx, yy);
      } else {
        kernel->gather(m.cart, m.offset, sx, yy);
      }
    }
  }
}

template <int ND> void Grid<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, gridLists, false, x, y); });
  if (vccLists) {
    Threads::StridedFor(vccLists.value().size(),
                        [&](Index const st, Index const sz) { forwardTask(st, sz, vccLists.value(), true, x, y); });
  }
  this->finishForward(y, time, false);
}

template <int ND> void Grid<ND>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, gridLists, false, x, y); });
  if (vccLists) {
    Threads::StridedFor(vccLists.value().size(),
                        [&](Index const st, Index const sz) { forwardTask(st, sz, vccLists.value(), true, x, y); });
  }
  this->finishForward(y, time, true);
}

template <int ND>
void Grid<ND>::adjointTask(Index const                   start,
                           Index const                   stride,
                           std::vector<CoordList> const &lists,
                           bool const                    isVCC,
                           CxNCMap<3> const             &y,
                           CxNMap<ND + 2>               &x) const

{
  Index const          nC = y.dimensions()[0];
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddFront(Constant<ND>(SubgridFullwidth(subgridW, kernel->paddedWidth())), nB, nC));
  Eigen::Tensor<Cx, 1> yy(nC);
  for (Index is = start; is < lists.size(); is += stride) {
    auto const &list = lists[is];
    sx.setZero();
    for (auto const &m : list.coords) {
      yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      if (basis) {
        kernel->spread(m.cart, m.offset, basis->entryConj(m.sample, m.trace), yy, sx);
      } else {
        kernel->spread(m.cart, m.offset, yy, sx);
      }
    }
    SubgridToGrid<ND>(mutexes, SubgridCorner(list.corner, subgridW, kernel->paddedWidth()), vccLists.has_value(), isVCC, sx, x);
  }
}

template <int ND> void Grid<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, gridLists, false, y, x); });
  if (vccLists) {
    Threads::StridedFor(vccLists.value().size(),
                        [&](Index const st, Index const sz) { adjointTask(st, sz, vccLists.value(), true, y, x); });
  }
  this->finishAdjoint(x, time, false);
}

template <int ND> void Grid<ND>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, gridLists, false, y, x); });
  if (vccLists) {
    Threads::StridedFor(vccLists.value().size(),
                        [&](Index const st, Index const sz) { adjointTask(st, sz, vccLists.value(), true, y, x); });
  }
  this->finishAdjoint(x, time, true);
}

template struct Grid<1>;
template struct Grid<2>;
template struct Grid<3>;

} // namespace TOps
} // namespace rl