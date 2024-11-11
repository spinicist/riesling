#pragma once

#include "basis/basis.hpp"
#include "kernel/kernel.hpp"
#include "top.hpp"
#include "trajectory.hpp"

#include <mutex>
#include <optional>

namespace rl {

template <int ND> struct GridOpts
{
  using Arrayf = Eigen::Array<float, ND, 1>;
  Arrayf      fov;
  Sz<ND>      matrix;
  float       osamp;
  std::string ktype;
  bool        vcc;
  Index       subgridSize;
};

template <int ND>
inline auto SubgridCorner(Eigen::Array<int16_t, ND, 1> const sgInd, Index const sgSz, Index const kW)
  -> Eigen::Array<int16_t, ND, 1>
{
  return (sgInd * sgSz) - (kW / 2);
}

inline auto SubgridFullwidth(Index const sgSize, Index const kW) { return sgSize + 2 * (kW / 2); }

namespace TOps {
template <int ND, bool VCC = false> struct Grid final : TOp<Cx, ND + 2 + VCC, 3>
{
  TOP_INHERIT(Cx, ND + 2 + VCC, 3)
  TOP_DECLARE(Grid)
  static auto Make(TrajectoryN<ND> const &t,
                   Sz<ND> const          &mat,
                   float const            os,
                   std::string const      kt,
                   Index const            nC,
                   Basis::CPtr            b,
                   Index const            sgW = 8) -> std::shared_ptr<Grid<ND, VCC>>;
  Grid(TrajectoryN<ND> const &traj,
       Sz<ND> const          &mat,
       float const            osamp,
       std::string const      ktype,
       Index const            nC,
       Basis::CPtr            b,
       Index const            sgW);
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

  std::shared_ptr<KernelBase<Scalar, ND>> kernel;

private:
  using CoordList = typename TrajectoryN<ND>::CoordList;
  Index                  subgridW;
  std::vector<CoordList> subs;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr                           basis;
  std::optional<std::vector<CoordList>> vccSubs;
};

} // namespace TOps
} // namespace rl
