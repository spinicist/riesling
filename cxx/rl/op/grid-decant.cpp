#include "grid-decant.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "grid-decant-subgrid.hpp"
#include "top-impl.hpp"

#include <numbers>

namespace rl {

namespace TOps {

template <int ND, typename KF>
GridDecant<ND, KF>::GridDecant(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &sk, Basis::CPtr b)
  : Parent(fmt::format("{}D Decant", ND))
  , kernel(opts.osamp)
  , subgridW{opts.subgridSize}
  , basis{b}
  , skern{sk}
{
  static_assert(ND < 4);
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  gridLists = traj.toCoordLists(osMatrix, kernel.FullWidth, subgridW, false);
  ishape = AddBack(osMatrix, basis ? basis->nB() : 1);
  oshape = Sz3{skern.dimension(1), traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  float const scale = std::sqrt(Product(LastN<3>(ishape)) / (float)Product(LastN<3>(skern.dimensions())));
  skern /= skern.constant(scale);

  Log::Print("Decant", "ishape {} oshape {} scale {}", this->ishape, this->oshape, scale);
}

template <int ND, typename KF>
auto GridDecant<ND, KF>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
  -> std::shared_ptr<GridDecant<ND>>
{
  return std::make_shared<GridDecant<ND>>(opts, t, skern, b);
}

template <int ND, typename KT>
void GridDecant<ND, KT>::forwardTask(Index const start, Index const stride, CxNCMap<ND + 1> const &x, CxNMap<3> &y) const
{
  Index const          nC = y.dimension(0);
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(subgridW, kernel.FullWidth)), nC, nB));
  Eigen::Tensor<Cx, 1> yy(Sz1{nC});
  Sz<ND + 2>           st, ksz;
  st.fill(0);
  ksz.fill(KT::FullWidth);
  ksz[ND - 1] = 1;
  ksz[ND - 2] = 1;
  for (Index is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    GridToDecant<ND>(SubgridCorner(list.corner, subgridW, kernel.FullWidth), skern, x, sx);
    for (auto const &m : list.coords) {
      yy.setZero();
      auto const k = kernel(m.offset);
      std::copy_n(m.cart.begin(), ND, st.begin());
      if (basis) {
        for (Index ib = 0; ib < basis->nB(); ib++) {
          auto const b = basis->entry(m.sample, m.trace, ib);
          st[ND - 1] = ib;
          for (Index ic = 0; ic < nC; ic++) {
            st[ND - 2] = ic;
            Cx0 const cc = (sx.slice(st, ksz) * k.template cast<Cx>() * b).sum();
            yy(ic) = cc();
          }
        }
      } else {
        for (Index ic = 0; ic < nC; ic++) {
          st[ND - 2] = ic;
          yy(ic) = Cx0((sx.slice(st, ksz) * k.template cast<Cx>()).sum())();
        }
      }
      y.template chip<2>(m.trace).template chip<1>(m.sample) += yy;
    }
  }
}

template <int ND, typename KT> void GridDecant<ND, KT>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Log::Print("FWD", "Forward size {}", gridLists.size());
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, x, y); });
  this->finishForward(y, time, false);
}

template <int ND, typename KT> void GridDecant<ND, KT>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, x, y); });
  this->finishForward(y, time, true);
}

template <int ND, typename KT>
void GridDecant<ND, KT>::adjointTask(Index const start, Index const stride, CxNCMap<3> const &y, CxNMap<ND + 1> &x) const

{
  Index const          nC = y.dimensions()[0];
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(subgridW, kernel.FullWidth)), nC, nB));
  Eigen::Tensor<Cx, 1> yy(nC);
  Sz<ND + 2>           st, ksz;
  st.fill(0);
  ksz.fill(KT::FullWidth);
  ksz[ND - 1] = 1;
  ksz[ND - 2] = 1;

  for (Index is = start; is < gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    sx.setZero();
    for (auto const &m : list.coords) {
      yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      auto const k = kernel(m.offset);
      std::copy_n(m.cart.begin(), ND, st.begin());
      if (basis) {
        for (Index ib = 0; ib < basis->nB(); ib++) {
          auto const b = std::conj(basis->entry(m.sample, m.trace, ib));
          st[ND - 1] = ib;
          for (Index ic = 0; ic < nC; ic++) {
            st[ND - 2] = ic;
            sx.slice(st, ksz) = k.template cast<Cx>() * b * yy(ic);
          }
        }
      } else {
        for (Index ic = 0; ic < nC; ic++) {
          st[ND - 2] = ic;
          sx.slice(st, ksz) = k.template cast<Cx>() * yy(ic);
        }
      }
      DecantToGrid<ND>(mutexes, SubgridCorner(list.corner, subgridW, kernel.FullWidth), skern, sx, x);
    }
  }
}

template <int ND, typename KT> void GridDecant<ND, KT>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KT> void GridDecant<ND, KT>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct GridDecant<1>;
template struct GridDecant<2>;
template struct GridDecant<3>;
} // namespace TOps
} // namespace rl