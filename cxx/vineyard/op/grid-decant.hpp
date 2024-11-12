#pragma once

#include "grid.hpp"

namespace rl {

namespace TOps {
template <int ND> struct GridDecant final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  TOP_DECLARE(GridDecant)

  GridDecant(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &traj, CxN<ND + 2> const &skern, Basis::CPtr b);
  static auto Make(Grid<ND>::Opts const &opts, TrajectoryN<ND> const &t, CxN<ND + 2> const &skern, Basis::CPtr b)
    -> std::shared_ptr<GridDecant<ND>>;

  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

  std::shared_ptr<KernelBase<Scalar, ND>> kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  Index                  subgridW;
  std::vector<CoordList> subs;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;
  CxN<ND + 2> skern;
};

} // namespace TOps
} // namespace rl
