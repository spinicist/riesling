#include "grid.hpp"

#include "grid-subgrid.hpp"
#include "log.hpp"
#include "op/top-impl.hpp"
#include "sys/threads.hpp"

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

template <int ND, bool VCC>
auto Grid<ND, VCC>::Make(TrajectoryN<ND> const &traj,
                         float const            osamp,
                         std::string const      ktype,
                         Index const            nC,
                         Basis::CPtr            b,
                         Index const            sgW) -> std::shared_ptr<Grid<ND, VCC>>
{
  return std::make_shared<Grid<ND, VCC>>(traj, osamp, ktype, nC, b, sgW);
}

template <bool VCC, int ND> auto AddVCC(Sz<ND> const cart, Index const nC, Index const nB) -> Sz<ND + 2 + VCC>
{
  if constexpr (VCC) {
    return AddFront(cart, nB, nC, 2);
  } else {
    return AddFront(cart, nB, nC);
  }
}

template <int ND, bool VCC>
Grid<ND, VCC>::Grid(TrajectoryN<ND> const &traj,
                    float const            osamp,
                    std::string const      ktype,
                    Index const            nC,
                    Basis::CPtr            b,
                    Index const            sgW)
  : Parent(fmt::format("{}D GridOp{}", ND, VCC ? " VCC" : ""))
  , kernel{KernelBase<Scalar, ND>::Make(ktype, osamp)}
  , subgridW{sgW}
  , basis{b}
{
  static_assert(ND < 4);
  auto const omatrix = MulToEven(traj.matrix(), osamp);
  mappings = CalcMapping(traj, omatrix, osamp, kernel->paddedWidth(), sgW);
  ishape = AddVCC<VCC>(omatrix, nC, basis ? basis->nB() : 1);
  oshape = Sz3{nC, traj.nSamples(), traj.nTraces()};
  mutexes = std::vector<std::mutex>(omatrix[ND - 1]);
  if constexpr (VCC) {
    Log::Print("Grid", "Adding VCC");
    auto const conjTraj = TrajectoryN<ND>(-traj.points(), traj.matrix(), traj.voxelSize());
    vccMapping = CalcMapping<ND>(conjTraj, omatrix, osamp, kernel->paddedWidth(), sgW);
  }
  Log::Debug("Grid", "ishape {} oshape {}", this->ishape, this->oshape);
}

/* Needs to be a functor to avoid template errors */
template <int ND, bool hasVCC, bool isVCC> struct forwardTask
{
  void operator()(Index const                     lo,
                  Index const                     hi,
                  std::vector<Mapping<ND>> const &mappings,
                  Index const                     sgW,
                  Basis::CPtr const              &basis,
                  KernelBase<Cx, ND>::Ptr const  &kernel,
                  CxNCMap<ND + 2 + hasVCC> const &x,
                  CxNMap<3>                      &y) const
  {
    Index const nC = y.dimension(0);
    Index const nB = basis ? basis->nB() : 1;
    CxN<ND + 2> sx(AddFront(Constant<ND>(SubgridFullwidth(sgW, kernel->paddedWidth())), nB, nC));
    auto        currentSg = mappings.front().subgrid;
    GridToSubgrid<ND, hasVCC, isVCC>(SubgridCorner(currentSg, sgW, kernel->paddedWidth()), x, sx);
    for (Index im = lo; im < hi; im++) {
      auto const &m = mappings[im];
      if ((currentSg != m.subgrid).any()) {
        currentSg = m.subgrid;
        GridToSubgrid<ND, hasVCC, isVCC>(SubgridCorner(currentSg, sgW, kernel->paddedWidth()), x, sx);
      }
      Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, m.sample, m.trace), Sz1{nC});
      if (basis) {
        kernel->gather(m.cart, m.offset, basis->entry(m.sample, m.trace), sx, yy);
      } else {
        kernel->gather(m.cart, m.offset, sx, yy);
      }
    }
  }
};

template <int ND, bool VCC> void Grid<ND, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::TensorDevice()) = y.constant(0.f);
  Threads::ChunkFor(forwardTask<ND, VCC, false>(), this->mappings, subgridW, this->basis, this->kernel, x, y);
  if constexpr (VCC == true) {
    Threads::ChunkFor(forwardTask<ND, VCC, true>(), this->vccMapping.value(), subgridW, this->basis, this->kernel, x, y);
  }
  this->finishForward(y, time, false);
}

template <int ND, bool VCC> void Grid<ND, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  Threads::ChunkFor(forwardTask<ND, VCC, false>(), this->mappings, subgridW, this->basis, this->kernel, x, y);
  if constexpr (VCC == true) {
    Threads::ChunkFor(forwardTask<ND, VCC, true>(), this->vccMapping.value(), subgridW, this->basis, this->kernel, x, y);
  }
  this->finishForward(y, time, true);
}

template <int ND, bool hasVCC, bool isVCC> struct adjointTask
{
  void operator()(Index const                     lo,
                  Index const                     hi,
                  std::vector<Mapping<ND>> const &mappings,
                  std::vector<std::mutex>        &mutexes,
                  Index const                     sgW,
                  Basis::CPtr const              &basis,
                  KernelBase<Cx, ND>::Ptr const  &kernel,
                  CxNCMap<3> const               &y,
                  CxNMap<ND + 2 + hasVCC>        &x) const
  {
    Index const          nC = y.dimensions()[0];
    Index const          nB = basis ? basis->nB() : 1;
    CxN<ND + 2>          sx(AddFront(Constant<ND>(SubgridFullwidth(sgW, kernel->paddedWidth())), nB, nC));
    Eigen::Tensor<Cx, 1> yy(nC);
    sx.setZero();
    auto currentSg = mappings[lo].subgrid;
    for (Index im = lo; im < hi; im++) {
      auto const &m = mappings[im];
      if ((currentSg != m.subgrid).any()) {
        SubgridToGrid<ND, hasVCC, isVCC>(mutexes, SubgridCorner(currentSg, sgW, kernel->paddedWidth()), sx, x);
        sx.setZero();
        currentSg = m.subgrid;
      }
      yy = y.template chip<2>(m.trace).template chip<1>(m.sample);
      if (basis) {
        kernel->spread(m.cart, m.offset, basis->entryConj(m.sample, m.trace), yy, sx);
      } else {
        kernel->spread(m.cart, m.offset, yy, sx);
      }
    }
    SubgridToGrid<ND, hasVCC, isVCC>(mutexes, SubgridCorner(currentSg, sgW, kernel->paddedWidth()), sx, x);
  }
};

template <int ND, bool VCC> void Grid<ND, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::TensorDevice()) = x.constant(0.f);
  Threads::ChunkFor(adjointTask<ND, VCC, false>(), this->mappings, mutexes, subgridW, this->basis, this->kernel, y, x);
  if constexpr (VCC == true) {
    Threads::ChunkFor(adjointTask<ND, VCC, true>(), this->vccMapping.value(), mutexes, subgridW, this->basis, this->kernel, y,
                      x);
  }
  this->finishAdjoint(x, time, false);
}

template <int ND, bool VCC> void Grid<ND, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, true);
  Threads::ChunkFor(adjointTask<ND, VCC, false>(), this->mappings, mutexes, subgridW, this->basis, this->kernel, y, x);
  if constexpr (VCC == true) {
    Threads::ChunkFor(adjointTask<ND, VCC, true>(), this->vccMapping.value(), mutexes, subgridW, this->basis, this->kernel, y,
                      x);
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