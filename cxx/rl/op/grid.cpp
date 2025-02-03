#include "grid.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "grid-subgrid.hpp"
#include "top-impl.hpp"

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
  ishape = AddBack(osMatrix, nC, basis ? basis->nB() : 1);
  oshape = Sz3{nC, traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  Log::Debug("Grid", "ishape {} oshape {}", this->ishape, this->oshape);
}

/* Needs to be a functor to avoid template errors */
template <int ND>
void Grid<ND>::forwardTask(Index const start, Index const stride, CxNCMap<ND + 2> const &x, CxNMap<3> &y) const
{
  Index const          nC = y.dimension(0);
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(subgridW, kernel->paddedWidth())), nC, nB));
  Eigen::Tensor<Cx, 1> yy(Sz1{nC});
  for (Index is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    GridToSubgrid<ND>(SubgridCorner(list.corner, subgridW, kernel->paddedWidth()), x, sx);
    for (auto const &m : list.coords) {
      yy.setZero();
      if (basis) {
        kernel->gather(m.cart, m.offset, basis->entry(m.sample, m.trace), sx, yy);
      } else {
        kernel->gather(m.cart, m.offset, sx, yy);
      }
      y.template chip<2>(m.trace).template chip<1>(m.sample) += yy;
    }
  }
}

template <int ND> void Grid<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, x, y); });
  this->finishForward(y, time, false);
}

template <int ND> void Grid<ND>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, x, y); });
  this->finishForward(y, time, true);
}

template <int ND>
void Grid<ND>::adjointTask(Index const start, Index const stride, CxNCMap<3> const &y, CxNMap<ND + 2> &x) const

{
  Index const          nC = y.dimensions()[0];
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(subgridW, kernel->paddedWidth())), nC, nB));
  Eigen::Tensor<Cx, 1> yy(nC);
  for (Index is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    sx.setZero();
    for (auto const &m : list.coords) {
      yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      if (basis) {
        kernel->spread(m.cart, m.offset, basis->entryConj(m.sample, m.trace), yy, sx);
      } else {
        kernel->spread(m.cart, m.offset, yy, sx);
      }
    }
    SubgridToGrid<ND>(mutexes, SubgridCorner(list.corner, subgridW, kernel->paddedWidth()), sx, x);
  }
}

template <int ND> void Grid<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND> void Grid<ND>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct Grid<1>;
template struct Grid<2>;
template struct Grid<3>;

} // namespace TOps
} // namespace rl