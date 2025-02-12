#pragma once

#include "grid.hpp"

namespace rl {

namespace TOps {
template <int ND_, typename KF = rl::ExpSemi<4>> struct GridDecant final : TOp<Cx, ND_ + 1, 3>
{
  static const int ND = ND_;
  using KType = Kernel<ND, KF>;

  TOP_INHERIT(Cx, ND + 1, 3)
  TOP_DECLARE(GridDecant)

  GridDecant(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr b);
  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
    -> std::shared_ptr<GridDecant<ND>>;

  void iforward(InCMap const x, OutMap y) const;
  void iadjoint(OutCMap const y, InMap x) const;

  KType kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  Index                  subgridW;
  std::vector<CoordList> gridLists;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;
  CxN<ND + 2> skern;

  void forwardTask(Index const start, Index const stride, CxNCMap<ND + 1> const &x, CxNMap<3> &y) const;
  void adjointTask(Index const start, Index const stride, CxNCMap<3> const &y, CxNMap<ND + 1> &x) const;
};

} // namespace TOps
} // namespace rl
