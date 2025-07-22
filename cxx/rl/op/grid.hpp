#pragma once

#include "../basis/basis.hpp"
#include "../kernel/kernel.hpp"
#include "../trajectory.hpp"
#include "grid-opts.hpp"
#include "top.hpp"

#include <mutex>
#include <optional>

namespace rl {

namespace TOps {
template <int ND_, typename KF = rl::ExpSemi<4>, int SGSZ_ = 8> struct Grid final : TOp<ND_ + 2, 3>
{
  static constexpr int ND = ND_;
  static constexpr int SGSZ = SGSZ_;
  static constexpr int SGFW = SGSZ + 2 * (KF::FullWidth / 2);
  using KType = Kernel<ND, KF>;

  TOP_INHERIT(ND + 2, 3)
  TOP_DECLARE(Grid)

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &t, Index const nC, Basis::CPtr b = nullptr)
    -> std::shared_ptr<Grid<ND, KF>>;
  Grid(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b = nullptr);
  void  iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void  iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  KType kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  std::vector<CoordList> gridLists;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;

  void forwardTask(Index const start, Index const stride, float const s, CxNCMap<ND + 2> const x, Cx3Map y) const;
  void adjointTask(Index const start, Index const stride, float const s, Cx3CMap y, CxNMap<ND + 2> x) const;
};

} // namespace TOps
} // namespace rl
