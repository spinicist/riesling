#pragma once

#include "grid.hpp"

namespace rl {

namespace TOps {
template <int ND> struct GridDecant final : TOp<Cx, ND + 1, 3>
{
  TOP_INHERIT(Cx, ND + 1, 3)
  TOP_DECLARE(GridDecant)
  static auto Make(TrajectoryN<ND> const &t,
                   Sz<ND> const          &matrix,
                   float const            os,
                   std::string const      kt,
                   CxN<ND + 2> const     &skern,
                   Basis::CPtr            b,
                   Index const            sgW = 32) -> std::shared_ptr<GridDecant<ND>>;
  GridDecant(TrajectoryN<ND> const &traj,
             Sz<ND> const          &matrix,
             float const            osamp,
             std::string const      ktype,
             CxN<ND + 2> const     &skern,
             Basis::CPtr            b,
             Index const            sgW);
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

  std::shared_ptr<KernelBase<Scalar, ND>> kernel;

private:
  Index                           subgridW;
  std::vector<SubgridMapping<ND>> subs;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr basis;
  CxN<ND + 2> skern;
};

} // namespace TOps
} // namespace rl
