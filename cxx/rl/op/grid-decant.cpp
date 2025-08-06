#include "grid-decant.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "grid-decant-subgrid.hpp"
#include "grid-func.hpp"
#include "top-impl.hpp"

namespace rl {

template <int ND> inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd, Index const sgSz, Index const kW)
  -> Eigen::Array<int16_t, ND, 1>
{
  return (sgInd * sgSz) - (kW / 2);
}

namespace TOps {

template <int ND, typename KF, int SG>
GridDecant<ND, KF, SG>::GridDecant(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &sk, Basis::CPtr b)
  : Parent(fmt::format("{}D Decant", ND))
  , kernel(opts.osamp)
  , basis{b}
  , skern{sk}
{
  static_assert(ND < 4);
  if (basis && basis->nB() != skern.dimension(4)) {
    throw Log::Failure(this->name, "Requires SENSE kernels to have same basis dimension as input");
  }
  auto const osMatrix = MulToEven(traj.matrixForFOV(opts.fov), opts.osamp);
  gridLists = traj.toCoordLists(osMatrix, kernel.FullWidth, SGSZ, false);
  ishape = AddBack(osMatrix, basis ? basis->nB() : 1);
  oshape = Sz3{skern.dimension(4), traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(osMatrix[ND - 1]);
  float const scale = std::sqrt(Product(FirstN<3>(ishape)) / (float)Product(FirstN<3>(skern.dimensions())));
  skern *= skern.constant(scale);
  Log::Print(this->name, "ishape {} oshape {} scale {}", this->ishape, this->oshape, scale);
}

template <int ND, typename KF, int SG>
auto GridDecant<ND, KF, SG>::Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
  -> std::shared_ptr<GridDecant<ND>>
{
  return std::make_shared<GridDecant<ND>>(opts, t, skern, b);
}

template <int ND, typename KF, int SG> void GridDecant<ND, KF, SG>::forwardTask(
  Index const start, Index const stride, float const s, CxNCMap<ND + 1> const &x, CxNMap<3> &y) const
{
  CxN<ND + 2> sx(AddBack(Constant<ND>(SGFW), basis ? basis->nB() : 1, y.dimension(0)));
  for (Index is = start; is < (Index)gridLists.size(); is += stride) {
    auto const &list = gridLists[is];
    auto const  corner = SubgridCorner<ND, SGSZ, KF::FullWidth>(list.corner);
    if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()), FirstN<ND>(skern.dimensions()))) {
      GridToDecant<ND, SGFW>::Fast(corner, skern, x, sx);
    } else {
      GridToDecant<ND, SGFW>::Slow(corner, skern, x, sx);
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

template <int ND, typename KF, int SG> void GridDecant<ND, KF, SG>::forward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, s, x, y); });
  this->finishForward(y, time, false);
}

template <int ND, typename KF, int SG> void GridDecant<ND, KF, SG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { forwardTask(st, sz, s, x, y); });
  this->finishForward(y, time, true);
}

template <int ND, typename KF, int SG> void GridDecant<ND, KF, SG>::adjointTask(
  Index const start, Index const stride, float const s, CxNCMap<3> const &y, CxNMap<ND + 1> &x) const

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
    if (InBounds<ND, SGFW>(corner, FirstN<ND>(x.dimensions()), FirstN<ND>(skern.dimensions()))) {
      DecantToGrid<ND, SGFW>::Fast(mutexes, corner, skern, sx, x);
    } else {
      DecantToGrid<ND, SGFW>::Slow(mutexes, corner, skern, sx, x);
    }
  }
}

template <int ND, typename KF, int SG> void GridDecant<ND, KF, SG>::adjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, s, y, x); });
  this->finishAdjoint(x, time, false);
}

template <int ND, typename KF, int SG> void GridDecant<ND, KF, SG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(gridLists.size(), [&](Index const st, Index const sz) { adjointTask(st, sz, s, y, x); });
  this->finishAdjoint(x, time, true);
}

template struct GridDecant<1>;
template struct GridDecant<2>;
template struct GridDecant<3>;
} // namespace TOps
} // namespace rl