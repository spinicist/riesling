#include "grid-decant.hpp"

#include "../log.hpp"
#include "../sys/threads.hpp"
#include "grid-decant-subgrid.hpp"
#include "top-impl.hpp"

#include <numbers>

namespace rl {

namespace TOps {

template <int ND, typename GT>
GridDecant<ND, GT>::GridDecant(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &sk, Basis::CPtr b)
  : Parent(fmt::format("{}D Decant", ND))
  , kernel(opts.osamp)
  , subgridW{opts.subgridSize}
  , basis{b}
  , skern{sk}
{
  static_assert(ND < 4);
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  subs = traj.toCoordLists(osMatrix, kernel.paddedWidth(), subgridW, false);
  ishape = AddBack(osMatrix, basis ? basis->nB() : 1);
  oshape = Sz3{skern.dimension(1), traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  float const scale = std::sqrt(Product(LastN<3>(ishape)) / (float)Product(LastN<3>(skern.dimensions())));
  skern /= skern.constant(scale);

  Log::Print("Decant", "ishape {} oshape {} scale {}", this->ishape, this->oshape, scale);
}

template <int ND, typename GT>
auto GridDecant<ND, GT>::Make(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
  -> std::shared_ptr<GridDecant<ND>>
{
  return std::make_shared<GridDecant<ND>>(opts, t, skern, b);
}

template <int ND, typename GT>
void GridDecant<ND, GT>::forwardTask(std::vector<typename TrajectoryN<ND>::CoordList> const &subs,
                                     Index const                                             start,
                                     Index const                                             stride,
                                     Index const                                             sgW,
                                     Basis::CPtr const                                      &basis,
                                     CxN<ND + 2> const                                      &skern,
                                     CxNCMap<ND + 1> const                                  &x,
                                     CxNMap<3>                                              &y) const
{
  Index const          nC = y.dimension(0);
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(sgW, kernel.paddedWidth())), nC, nB));
  Eigen::Tensor<Cx, 1> yy(Sz1{nC});
  for (Index is = start; is < subs.size(); is += stride) {
    auto const &sub = subs[is];
    GridToDecant<ND>(SubgridCorner(sub.corner, sgW, kernel.paddedWidth()), skern, x, sx);
    for (auto const &m : sub.coords) {
      yy.setZero();
      if (basis) {
        kernel.gather(m.cart, m.offset, basis->entry(m.sample, m.trace), sx, yy);
      } else {
        kernel.gather(m.cart, m.offset, sx, yy);
      }
      y.template chip<2>(m.trace).template chip<1>(m.sample) += yy;
    }
  }
}

template <int ND, typename GT> void GridDecant<ND, GT>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::StridedFor(subs.size(),
                      [&](Index const st, Index const sz) { forwardTask(subs, st, sz, subgridW, basis, skern, x, y); });
  this->finishForward(y, time, false);
}

template <int ND, typename GT> void GridDecant<ND, GT>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(subs.size(),
                      [&](Index const st, Index const sz) { forwardTask(subs, st, sz, subgridW, basis, skern, x, y); });
  this->finishForward(y, time, true);
}

template <int ND, typename GT>
void GridDecant<ND, GT>::adjointTask(std::vector<typename TrajectoryN<ND>::CoordList> const &subs,
                                     Index const                                             start,
                                     Index const                                             stride,
                                     std::vector<std::mutex>                                &mutexes,
                                     Index const                                             sgW,
                                     Basis::CPtr const                                      &basis,
                                     CxN<ND + 2> const                                      &skern,
                                     CxNCMap<3> const                                       &y,
                                     CxNMap<ND + 1>                                         &x) const
{
  Index const          nC = y.dimensions()[0];
  Index const          nB = basis ? basis->nB() : 1;
  CxN<ND + 2>          sx(AddBack(Constant<ND>(SubgridFullwidth(sgW, kernel.paddedWidth())), nC, nB));
  Eigen::Tensor<Cx, 1> yy(nC);
  for (Index is = start; is < subs.size(); is += stride) {
    auto const &sub = subs[is];
    sx.setZero();
    for (auto const &m : sub.coords) {
      yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      if (basis) {
        kernel.spread(m.cart, m.offset, basis->entryConj(m.sample, m.trace), yy, sx);
      } else {
        kernel.spread(m.cart, m.offset, yy, sx);
      }
    }
    DecantToGrid<ND>(mutexes, SubgridCorner(sub.corner, sgW, kernel.paddedWidth()), skern, sx, x);
  }
}

template <int ND, typename GT> void GridDecant<ND, GT>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(
    subs.size(), [&](Index const st, Index const sz) { adjointTask(subs, st, sz, mutexes, subgridW, basis, skern, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND, typename GT> void GridDecant<ND, GT>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(
    subs.size(), [&](Index const st, Index const sz) { adjointTask(subs, st, sz, mutexes, subgridW, basis, skern, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct GridDecant<1>;
template struct GridDecant<2>;
template struct GridDecant<3>;
} // namespace TOps
} // namespace rl