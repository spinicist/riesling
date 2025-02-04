#pragma once

#include "grid.hpp"

namespace rl {

namespace TOps {
template <int ND_, typename GType = Kernel<Cx, ND_, rl::ExpSemi<4>>> struct GridDecant final : TOp<Cx, ND_ + 1, 3>
{
  TOP_INHERIT(Cx, ND_ + 1, 3)
  TOP_DECLARE(GridDecant)
  static const int ND = ND_;

  GridDecant(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr b);
  static auto Make(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
    -> std::shared_ptr<GridDecant<ND>>;

  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

  GType kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  Index                  subgridW;
  std::vector<CoordList> subs;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;
  CxN<ND + 2> skern;

  void forwardTask(std::vector<typename TrajectoryN<ND>::CoordList> const &subs,
                   Index const                                             start,
                   Index const                                             stride,
                   Index const                                             sgW,
                   Basis::CPtr const                                      &basis,
                   CxN<ND + 2> const                                      &skern,
                   CxNCMap<ND + 1> const                                  &x,
                   CxNMap<3>                                              &y) const;
  void adjointTask(std::vector<typename TrajectoryN<ND>::CoordList> const &subs,
                   Index const                                             start,
                   Index const                                             stride,
                   std::vector<std::mutex>                                &mutexes,
                   Index const                                             sgW,
                   Basis::CPtr const                                      &basis,
                   CxN<ND + 2> const                                      &skern,
                   CxNCMap<3> const                                       &y,
                   CxNMap<ND + 1>                                         &x) const;
};

} // namespace TOps
} // namespace rl
