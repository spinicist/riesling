#include "grid.hpp"

#include "threads.hpp"
#include "top.hpp"

#include <mutex>
#include <numbers>

namespace {
constexpr float inv_sqrt2 = std::numbers::sqrt2 / 2;
}

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
  TrajectoryN<NDim> const &traj, std::string const ktype, float const osamp, Index const nC, Basis const &b, Index const bSz)
  -> std::shared_ptr<Grid<NDim, VCC>>
{
  return std::make_shared<Grid<NDim, VCC>>(traj, ktype, osamp, nC, b, bSz);
}

template <bool VCC, int ND> auto AddVCC(Sz<ND> const cart, Index const nC, Index const nB) -> Sz<ND + 2 + VCC>
{
  if constexpr (VCC) {
    return AddFront(cart, nC, 2, nB);
  } else {
    return AddFront(cart, nC, nB);
  }
}

template <int NDim, bool VCC>
Grid<NDim, VCC>::Grid(
  TrajectoryN<NDim> const &traj, std::string const ktype, float const osamp, Index const nC, Basis const &b, Index const bSz)
  : Parent(fmt::format("{}D GridOp{}", NDim, VCC ? " VCC" : ""))
  , kernel{Kernel<Scalar, NDim>::Make(ktype, osamp)}
  , basis{b}
{
  static_assert(NDim < 4);

  auto const m = CalcMapping(traj, osamp, kernel->paddedWidth(), bSz);
  mapping = m.mappings;
  ishape = AddVCC<VCC>(m.cartDims, nC, b.dimension(0));
  oshape = AddFront(m.noncartDims, nC);
  if constexpr (VCC) {
    Log::Print("Adding VCC");
    auto const conjTraj = TrajectoryN<NDim>(-traj.points(), traj.matrix(), traj.voxelSize());
    vccMapping = CalcMapping<NDim>(conjTraj, osamp, kernel->paddedWidth(), bSz).mappings;
  }
  Log::Debug("Grid Dims {}", this->ishape);
}

template <int ND, bool hasVCC, bool isVCC>
inline void forwardTask(std::vector<Mapping<ND>> const                                   &mappings,
                        Basis const                                                      &basis,
                        std::shared_ptr<Kernel<Cx, ND>> const                            &kernel,
                        Eigen::TensorMap<Eigen::Tensor<Cx, ND + 2 + hasVCC> const> const &x,
                        Eigen::TensorMap<Eigen::Tensor<Cx, 3>>                           &y)
{
  Index const nC = y.dimension(0);
  Index const nB = basis.dimension(0);
  // auto        grid_task = [&](Index const is) {
  CxN<ND + 2> sx(AddFront(mappings.front().subgrid.size(), nC, nB));
  sx.setZero();
  Sz<ND>      current = mappings.front().subgrid.minCorner;
  mappings.front().subgrid.template gridToSubgrid<hasVCC, isVCC>(x, sx);
  for (auto const &m : mappings) {
    if (current != m.subgrid.minCorner) {
      m.subgrid.template gridToSubgrid<hasVCC, isVCC>(x, sx);
      sx.resize(AddFront(m.subgrid.size(), nC, nB));
      sx.setZero();
      current = m.subgrid.minCorner;
    }

    Eigen::Tensor<Cx, 1> const bs =
      basis.template chip<2>(m.noncart.trace % basis.dimension(2)).template chip<1>(m.noncart.sample % basis.dimension(1));
    Eigen::TensorMap<Eigen::Tensor<Cx, 1>> yy(&y(0, m.noncart.sample, m.noncart.trace), Sz1{nC});
    kernel->gather(m.cart, m.offset, m.subgrid.minCorner, bs, sx, yy);
  }
}
// };
// Threads::For(grid_task, map.subgrids.size());

template <int NDim, bool VCC> void Grid<NDim, VCC>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, false);
  y.device(Threads::GlobalDevice()) = y.constant(0.f);
  forwardTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, x, y);
  if (this->vccMapping) { forwardTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, x, y); }
  this->finishForward(y, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iforward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x, y, true);
  forwardTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, x, y);
  if (this->vccMapping) { forwardTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, x, y); }
  this->finishForward(y, time, true);
}

template <int ND, bool hasVCC, bool isVCC>
inline void adjointTask(std::vector<Mapping<ND>> const                       &mappings,
                        Basis const                                          &basis,
                        std::shared_ptr<Kernel<Cx, ND>> const                &kernel,
                        Eigen::TensorMap<Eigen::Tensor<Cx, 3> const> const   &y,
                        Eigen::TensorMap<Eigen::Tensor<Cx, ND + 2 + hasVCC>> &x)
{
  Index const nC = y.dimensions()[0];
  Index const nB = basis.dimension(0);

  std::mutex writeMutex;
  // auto       grid_task = [&](Index is) {

  CxN<ND + 2> sx(AddFront(mappings.front().subgrid.size(), nC, nB));
  sx.setZero();
  Sz<ND>      current = mappings.front().subgrid.minCorner;

  for (auto const &m : mappings) {
    if (current != m.subgrid.minCorner) {
      fmt::print(stderr, "current {} next {} size {}\n", current, m.subgrid.minCorner, m.subgrid.size());
      std::scoped_lock lock(writeMutex);
      m.subgrid.template subgridToGrid<hasVCC, isVCC>(sx, x);

      sx.resize(AddFront(m.subgrid.size(), nC, nB));
      sx.setZero();
      current = m.subgrid.minCorner;
    }

    Eigen::Tensor<Cx, 1> const bs = basis.template chip<2>(m.noncart.trace % basis.dimension(2))
                                      .template chip<1>(m.noncart.sample % basis.dimension(1))
                                      .conjugate();
    Eigen::Tensor<Cx, 1> yy = y.template chip<2>(m.noncart.trace).template chip<1>(m.noncart.sample);
    kernel->spread(m.cart, m.offset, m.subgrid.minCorner, bs, yy, sx);
  }

  {
    std::scoped_lock lock(writeMutex);
    mappings.back().subgrid.template subgridToGrid<hasVCC, isVCC>(sx, x);
  }
}
// Threads::For(grid_task, map.subgrids.size());

template <int NDim, bool VCC> void Grid<NDim, VCC>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.device(Threads::GlobalDevice()) = x.constant(0.f);
  adjointTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, y, x);
  if (this->vccMapping) { adjointTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, y, x); }
  this->finishAdjoint(x, time, false);
}

template <int NDim, bool VCC> void Grid<NDim, VCC>::iadjoint(OutCMap const &y, InMap &x) const
{

  auto const time = this->startAdjoint(y, x, true);
  adjointTask<NDim, VCC, false>(this->mapping, this->basis, this->kernel, y, x);
  if (this->vccMapping) { adjointTask<NDim, VCC, true>(this->vccMapping.value(), this->basis, this->kernel, y, x); }
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