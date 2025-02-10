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

namespace TOps {
template <int ND_, typename KType_ = Kernel<ND_, rl::ExpSemi<4>>> struct Grid final : TOp<Cx, ND_ + 2, 3>
{
  static const int ND = ND_;
  using KType = KType_;

  TOP_INHERIT(Cx, ND + 2, 3)
  TOP_DECLARE(Grid)

  struct Opts
  {
    using Arrayf = Eigen::Array<float, ND, 1>;
    Arrayf fov = Arrayf::Zero();
    float  osamp = 1.3f;
    Index  subgridSize = 8;
  };

  static auto Make(Opts const &opts, TrajectoryN<ND> const &t, Index const nC, Basis::CPtr b)
    -> std::shared_ptr<Grid<ND, KType>>;
  Grid(Opts const &opts, TrajectoryN<ND> const &traj, Index const nC, Basis::CPtr b);
  void  iforward(InCMap const &x, OutMap &y) const;
  void  iadjoint(OutCMap const &y, InMap &x) const;
  KType kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  Index                  subgridW;
  std::vector<CoordList> gridLists;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;

  void forwardTask(Index const start, Index const stride, CxNCMap<ND + 2> const &x, CxNMap<3> &y) const;
  void adjointTask(Index const start, Index const stride, CxNCMap<3> const &y, CxNMap<ND + 2> &x) const;
};

} // namespace TOps
} // namespace rl
