#pragma once

#include "../basis/basis.hpp"
#include "../kernel/kernel.hpp"
#include "../trajectory.hpp"
#include "top.hpp"

#include <mutex>
#include <optional>

namespace rl {

template <int ND>
inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd, Index const sgSz, Index const kW)
  -> Eigen::Array<int16_t, ND, 1>
{
  return (sgInd * sgSz) - (kW / 2);
}

inline auto SubgridFullwidth(Index const sgSize, Index const kW) { return sgSize + 2 * (kW / 2); }

template <int ND> struct GridOpts
{
  using Arrayf = Eigen::Array<float, ND, 1>;
  Arrayf fov = Arrayf::Zero();
  float  osamp = 1.3f;
};

namespace TOps {
template <int ND_, typename KF = rl::ExpSemi<4>, int SGSZ_ = 4> struct Grid final : TOp<Cx, ND_ + 2, 3>
{
  static constexpr int ND = ND_;
  static constexpr int SGSZ = SGSZ_;
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

  void forwardTask(Index const start, Index const stride, CxNCMap<ND + 2> const &x, CxNMap<3> &y) const;
  void adjointTask(Index const start, Index const stride, CxNCMap<3> const &y, CxNMap<ND + 2> &x) const;
};

} // namespace TOps
} // namespace rl
