#include "grid.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "grid-subgrid.hpp"
#include "top-impl.hpp"

#include <numbers>

namespace rl {

namespace TOps {

template <int ND, typename KF>
auto Grid<ND, KF>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  -> std::shared_ptr<Grid<ND, KF>>
{
  return std::make_shared<Grid<ND, KF>>(opts, traj, nC, b);
}

template <int ND, typename KF>
Grid<ND, KF>::Grid(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b)
  : Parent(fmt::format("Grid{}D", ND))
  , kernel(opts.osamp)
  , subgridW{opts.subgridSize}
  , basis{b}
{
  static_assert(ND < 4);
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  gridLists = traj.toCoordLists(osMatrix, kernel.FullWidth, subgridW, false);
  ishape = AddBack(osMatrix, nC, basis ? basis->nB() : 1);
  oshape = Sz3{nC, traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  Log::Debug("Grid", "ishape {} oshape {}", this->ishape, this->oshape);
}

template <int ND, typename KF>
void Grid<ND, KF>::forwardTask(Index const start, Index const stride, CxNCMap<ND + 2> const &x, CxNMap<3> &y) const
{
  Index const          nC = y.dimension(0);
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(subgridW, kernel.FullWidth)), nC, nB));
  Eigen::Tensor<Cx, 1> yy(Sz1{nC});
  Sz<ND>               st, sz;
  st.fill(0);
  sz.fill(KF::FullWidth);
  for (Index is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    GridToSubgrid<ND>(SubgridCorner(list.corner, subgridW, kernel.FullWidth), x, sx);
    for (auto const &m : list.coords) {
      yy.setZero();
      auto const k = kernel(m.offset);
      std::transform(m.cart.begin(), m.cart.end(), st.begin(), [a=kernel.FullWidth/2](auto const i) { return i - a; });
      if (basis) {
        for (Index ib = 0; ib < basis->nB(); ib++) {
          auto const b = basis->entry(ib, m.sample, m.trace);
          for (Index ic = 0; ic < nC; ic++) {
            Cx0 const cc =
              (sx.template chip<ND + 1>(ib).template chip<ND>(ic).slice(st, sz) * k.template cast<Cx>() * b).sum();
            yy(ic) += cc();
          }
        }
      } else {
        for (Index ic = 0; ic < nC; ic++) {
          yy(ic) = Cx0((sx.template chip<ND + 1>(0).template chip<ND>(ic).slice(st, sz) * k.template cast<Cx>()).sum())();
        }
      }
      y.template chip<2>(m.trace).template chip<1>(m.sample) += yy;
    }
  }
}

template <int ND, typename KF> void Grid<ND, KF>::forward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Log::Print("FWD", "Forward size {}", gridLists.size());
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, x, y); });
  this->finishForward(y, time, false);
}

template <int ND, typename KF> void Grid<ND, KF>::iforward(InCMap const x, OutMap y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, x, y); });
  this->finishForward(y, time, true);
}

template <int ND, typename KF>
void Grid<ND, KF>::adjointTask(Index const start, Index const stride, CxNCMap<3> const &y, CxNMap<ND + 2> &x) const

{
  Index const          nC = y.dimensions()[0];
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(subgridW, kernel.FullWidth)), nC, nB));
  Eigen::Tensor<Cx, 1> yy(nC);
  Sz<ND>               st, sz;
  st.fill(0);
  sz.fill(KF::FullWidth);
  for (Index is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    sx.setZero();
    for (auto const &m : list.coords) {
      yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      auto const k = kernel(m.offset);
      std::transform(m.cart.begin(), m.cart.end(), st.begin(), [a=kernel.FullWidth/2](auto const i) { return i - a; });
      if (basis) {
        for (Index ib = 0; ib < basis->nB(); ib++) {
          auto const b = std::conj(basis->entry(ib, m.sample, m.trace));
          for (Index ic = 0; ic < nC; ic++) {
            sx.template chip<ND + 1>(ib).template chip<ND>(ic).slice(st, sz) += k.template cast<Cx>() * b * yy(ic);
          }
        }
      } else {
        for (Index ic = 0; ic < nC; ic++) {
          sx.template chip<ND + 1>(0).template chip<ND>(ic).slice(st, sz) += k.template cast<Cx>() * yy(ic);
        }
      }
    }
    SubgridToGrid<ND>(mutexes, SubgridCorner(list.corner, subgridW, kernel.FullWidth), sx, x);
  }
}

template <int ND, typename KF> void Grid<ND, KF>::adjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF> void Grid<ND, KF>::iadjoint(OutCMap const y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct Grid<1, rl::ExpSemi<4>>;
template struct Grid<2, rl::ExpSemi<4>>;
template struct Grid<3, rl::ExpSemi<4>>;

template struct Grid<1, rl::ExpSemi<6>>;
template struct Grid<2, rl::ExpSemi<6>>;
template struct Grid<3, rl::ExpSemi<6>>;

template struct Grid<1, rl::TopHat<1>>;
template struct Grid<2, rl::TopHat<1>>;
template struct Grid<3, rl::TopHat<1>>;

} // namespace TOps
} // namespace rl