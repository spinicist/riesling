#include "grid-decant.hpp"

#include "grid-decant-subgrid.hpp"
#include "log.hpp"
#include "op/top-impl.hpp"
#include "sys/threads.hpp"

#include <numbers>

namespace rl {

namespace TOps {

template <int ND>
auto GridDecant<ND>::Make(TrajectoryN<ND> const &traj,
                          Sz<ND> const          &matrix,
                          float const            osamp,
                          std::string const      ktype,
                          CxN<ND + 2> const     &sk,
                          Basis::CPtr            b,
                          Index const            sgW) -> std::shared_ptr<GridDecant<ND>>
{
  return std::make_shared<GridDecant<ND>>(traj, matrix, osamp, ktype, sk, b, sgW);
}

template <int ND>
GridDecant<ND>::GridDecant(TrajectoryN<ND> const &traj,
                           Sz<ND> const          &matrix,
                           float const            osamp,
                           std::string const      ktype,
                           CxN<ND + 2> const     &sk,
                           Basis::CPtr            b,
                           Index const            sgW)
  : Parent(fmt::format("{}D Decant", ND))
  , kernel{KernelBase<Scalar, ND>::Make(ktype, osamp)}
  , subgridW{sgW}
  , basis{b}
  , skern{sk}
{
  static_assert(ND < 4);
  auto const omatrix = MulToEven(matrix, osamp);
  subs = CalcMapping(traj, omatrix, kernel->paddedWidth(), sgW);
  ishape = AddFront(omatrix, basis ? basis->nB() : 1);
  oshape = Sz3{skern.dimension(1), traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(omatrix[ND - 1]);
  float const scale = std::sqrt(Product(LastN<3>(ishape)) / (float)Product(LastN<3>(skern.dimensions())));
  skern /= skern.constant(scale);

  Log::Print("Decant", "ishape {} oshape {} scale {}", this->ishape, this->oshape, scale);
}

/* Needs to be a functor to avoid template errors */
template <int ND> struct forwardTask
{
  void operator()(std::vector<SubgridMapping<ND>> const &subs,
                  Index const                            start,
                  Index const                            stride,
                  Index const                            sgW,
                  Basis::CPtr const                     &basis,
                  KernelBase<Cx, ND>::Ptr const         &kernel,
                  CxN<ND + 2> const                     &skern,
                  CxNCMap<ND + 1> const                 &x,
                  CxNMap<3>                             &y) const
  {
    Index const nC = y.dimension(0);
    Index const nB = basis ? basis->nB() : 1;
    CxN<ND + 2> sx(AddFront(Constant<ND>(SubgridFullwidth(sgW, kernel->paddedWidth())), nB, nC));
    for (Index is = start; is < subs.size(); is += stride) {
      auto const &sub = subs[is];
      GridToDecant<ND>(SubgridCorner(sub.corner, sgW, kernel->paddedWidth()), skern, x, sx);
      for (auto const &m : sub.mappings) {
        Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, m.sample, m.trace), Sz1{nC});
        if (basis) {
          kernel->gather(m.cart, m.offset, basis->entry(m.sample, m.trace), sx, yy);
        } else {
          kernel->gather(m.cart, m.offset, sx, yy);
        }
      }
    }
  }
};

template <int ND> void GridDecant<ND>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::StridedFor(forwardTask<ND>(), this->subs, subgridW, this->basis, this->kernel, this->skern, x, y);
  this->finishForward(y, time, false);
}

template <int ND> void GridDecant<ND>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::StridedFor(forwardTask<ND>(), this->subs, subgridW, this->basis, this->kernel, this->skern, x, y);
  this->finishForward(y, time, true);
}

template <int ND> struct adjointTask
{
  void operator()(std::vector<SubgridMapping<ND>> const &subs,
                  Index const                            start,
                  Index const                            stride,
                  std::vector<std::mutex>               &mutexes,
                  Index const                            sgW,
                  Basis::CPtr const                     &basis,
                  KernelBase<Cx, ND>::Ptr const         &kernel,
                  CxN<ND + 2> const                     &skern,
                  CxNCMap<3> const                      &y,
                  CxNMap<ND + 1>                        &x) const
  {
    Index const          nC = y.dimensions()[0];
    Index const          nB = basis ? basis->nB() : 1;
    CxN<ND + 2>          sx(AddFront(Constant<ND>(SubgridFullwidth(sgW, kernel->paddedWidth())), nB, nC));
    Eigen::Tensor<Cx, 1> yy(nC);
    for (Index is = start; is < subs.size(); is += stride) {
      auto const &sub = subs[is];
      sx.setZero();
      for (auto const &m : sub.mappings) {
        yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
        if (basis) {
          kernel->spread(m.cart, m.offset, basis->entryConj(m.sample, m.trace), yy, sx);
        } else {
          kernel->spread(m.cart, m.offset, yy, sx);
        }
      }
      DecantToGrid<ND>(mutexes, SubgridCorner(sub.corner, sgW, kernel->paddedWidth()), skern, sx, x);
    }
  }
};

template <int ND> void GridDecant<ND>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::StridedFor(adjointTask<ND>(), this->subs, mutexes, subgridW, this->basis, this->kernel, this->skern, y, x);
  this->finishAdjoint(x, time, false);
}

template <int ND> void GridDecant<ND>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::StridedFor(adjointTask<ND>(), this->subs, mutexes, subgridW, this->basis, this->kernel, this->skern, y, x);
  this->finishAdjoint(x, time, true);
}

template struct GridDecant<1>;
template struct GridDecant<2>;
template struct GridDecant<3>;
} // namespace TOps
} // namespace rl