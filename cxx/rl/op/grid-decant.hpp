#pragma once

#include "grid.hpp"

namespace rl {

namespace TOps {
template <int ND_, typename KF = rl::ExpSemi<4>, int SGSZ_ = 8> struct GridDecant final : TOp<ND_ + 1, 3>
{
  static constexpr int ND = ND_;
  static constexpr int SGSZ = SGSZ_;
  static constexpr int SGFW = SGSZ + 2 * (KF::FullWidth / 2);
  using KType = Kernel<ND, KF>;

  TOP_INHERIT(ND + 1, 3)
  TOP_DECLARE(GridDecant)

  GridDecant(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr b);
  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
    -> std::shared_ptr<GridDecant<ND>>;

  void iforward(InCMap x, OutMap y, float const s) const;
  void iadjoint(OutCMap y, InMap x, float const s) const;

  KType kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  std::vector<CoordList> gridLists;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;
  CxN<ND + 2> skern;

  void forwardTask(Index const start, Index const stride, float const s, CxNCMap<ND + 1> const &x, CxNMap<3> &y) const;
  void adjointTask(Index const start, Index const stride, float const s, CxNCMap<3> const &y, CxNMap<ND + 1> &x) const;
};

} // namespace TOps
} // namespace rl
