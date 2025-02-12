#pragma once

#include "../basis/basis.hpp"
#include "../kernel/kernel.hpp"
#include "../trajectory.hpp"
#include "grid-opts.hpp"
#include "top.hpp"

#include <mutex>
#include <optional>

namespace rl {

template <int ND, int SGSZ, int KW>
inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd) -> Eigen::Array<int16_t, ND, 1>
{
  return (sgInd * SGSZ) - (KW / 2);
}

template <int ND, int SGSZ> inline auto InBounds(Eigen::Array<int16_t, ND, 1> const corner, Sz<ND> const gSz)
{
  bool inBounds = true;
  for (Index ii = 0; ii < ND; ii++) {
    if (corner[ii] < 0 || (corner[ii] + SGSZ >= gSz[ii])) { inBounds = false; }
  }
  return inBounds;
}

namespace TOps {
template <int ND_, typename KF = rl::ExpSemi<4>, int SGSZ_ = 4> struct Grid final : TOp<Cx, ND_ + 2, 3>
{
  static constexpr int ND = ND_;
  static constexpr int SGSZ = SGSZ_;
  static constexpr int SGFW = SGSZ + 2 * (KF::FullWidth / 2);
  using KType = Kernel<ND, KF>;

  TOP_INHERIT(Cx, ND + 2, 3)
  TOP_DECLARE(Grid)

  static auto Make(GridOpts<ND> const &opts, TrajectoryN<ND> const &t, Index const nC, Basis::CPtr b)
    -> std::shared_ptr<Grid<ND, KF>>;
  Grid(GridOpts<ND> const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b);
  void  iforward(InCMap const x, OutMap y) const;
  void  iadjoint(OutCMap const y, InMap x) const;
  KType kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  std::vector<CoordList> gridLists;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;

  void forwardTask(Index const start, Index const stride, CxNCMap<ND + 2> const x, Cx3Map y) const;
  void adjointTask(Index const start, Index const stride, Cx3CMap const y, CxNMap<ND + 2> x) const;
};

} // namespace TOps
} // namespace rl
