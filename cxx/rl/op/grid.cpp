#include "grid.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "grid-func.hpp"
#include "grid-subgrid.hpp"
#include "top-impl.hpp"

namespace rl {

namespace TOps {

template <int ND, typename KF, int SG>
auto Grid<ND, KF, SG>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  -> std::shared_ptr<Grid<ND, KF>>
{
  return std::make_shared<Grid<ND, KF>>(opts, traj, nC, b);
}

template <int ND, typename KF, int SG>
Grid<ND, KF, SG>::Grid(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  : Parent(fmt::format("Grid{}D", ND))
  , kernel(opts.osamp)
  , basis{b}
{
  static_assert(ND < 4);
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  gridLists = traj.toCoordLists(osMatrix, kernel.FullWidth, SGSZ, false);
  ishape = AddBack(osMatrix, basis ? basis->nB() : 1, nC);
  oshape = Sz3{nC, traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  Log::Debug("Grid", "ishape {} oshape {}", this->ishape, this->oshape);
}

template <int ND, typename KF, int SG> void
Grid<ND, KF, SG>::forwardTask(Index const start, Index const stride, float const s, CxNCMap<ND + 2> const x, Cx3Map y) const
{
  CxN<ND + 2> sx(AddBack(Constant<ND>(SGFW), basis ? basis->nB() : 1, y.dimension(0)));
  for (size_t is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    auto const  corner = SubgridCorner<ND, SGSZ, KF::FullWidth>(list.corner);
    if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()))) {
      GridToSubgrid<ND, SGFW>::FastCopy(corner, x, sx);
    } else {
      GridToSubgrid<ND, SGFW>::SlowCopy(corner, x, sx);
    }
    for (auto const &m : list.coords) {
      typename KType::Tensor const k = kernel(m.offset) * s;
      if (basis) {
        GFunc<ND, KF::FullWidth>::Gather(basis, m.cart, m.sample, m.trace, k, sx, y);
      } else {
        GFunc<ND, KF::FullWidth>::Gather(m.cart, m.sample, m.trace, k, sx, y);
      }
    }
  }
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, s, x, y); });
  this->finishForward(y, time, false);
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, s, x, y); });
  this->finishForward(y, time, true);
}

template <int ND, typename KF, int SG>
void Grid<ND, KF, SG>::adjointTask(Index const start, Index const stride, float const s, Cx3CMap y, CxNMap<ND + 2> x) const

{
  CxN<ND + 2> sx(AddBack(Constant<ND>(SGFW), basis ? basis->nB() : 1, y.dimension(0)));
  for (size_t is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    sx.setZero();
    for (auto const &m : list.coords) {
      typename KType::Tensor const k = kernel(m.offset) * s;
      if (basis) {
        GFunc<ND, KF::FullWidth>::Scatter(basis, m.cart, m.sample, m.trace, k, y, sx);
      } else {
        GFunc<ND, KF::FullWidth>::Scatter(m.cart, m.sample, m.trace, k, y, sx);
      }
    }
    auto const corner = SubgridCorner<ND, SGSZ, KF::FullWidth>(list.corner);
    if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()))) {
      SubgridToGrid<ND, SGFW>::FastCopy(mutexes, corner, sx, x);
    } else {
      SubgridToGrid<ND, SGFW>::SlowCopy(mutexes, corner, sx, x);
    }
  }
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, s, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF, int SG> void Grid<ND, KF, SG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, s, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct Grid<1, rl::ExpSemi<4>>;
template struct Grid<2, rl::ExpSemi<4>>;
template struct Grid<3, rl::ExpSemi<4>>;

template struct Grid<1, rl::ExpSemi<6>>;
template struct Grid<2, rl::ExpSemi<6>>;
template struct Grid<3, rl::ExpSemi<6>>;

template struct Grid<1, rl::ExpSemi<8>>;
template struct Grid<2, rl::ExpSemi<8>>;
template struct Grid<3, rl::ExpSemi<8>>;

template struct Grid<1, rl::TopHat<1>>;
template struct Grid<2, rl::TopHat<1>>;
template struct Grid<3, rl::TopHat<1>>;

} // namespace TOps
} // namespace rl