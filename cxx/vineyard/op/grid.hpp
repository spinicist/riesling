#pragma once

#include "basis/basis.hpp"
#include "grid-mapping.hpp"
#include "kernel/kernel.hpp"
#include "top.hpp"

#include <args.hxx>

namespace rl {

struct GridOpts
{
  GridOpts(args::Subparser &parser);
  args::ValueFlag<std::string> ktype;
  args::ValueFlag<float>       osamp;
  args::Flag                   vcc;
  args::ValueFlag<Index>       batches, subgridSize;
};

namespace TOps {
template <int ND, bool VCC = false> struct Grid final : TOp<Cx, ND + 2 + VCC, 3>
{
  TOP_INHERIT(Cx, ND + 2 + VCC, 3)
  using Parent::adjoint;
  using Parent::forward;
  std::shared_ptr<Kernel<Scalar, ND>>     kernel;
  Index                                   subgridW;
  std::vector<Mapping<ND>>                mappings;
  Basis                                   basis;
  std::optional<std::vector<Mapping<ND>>> vccMapping;

  static auto Make(TrajectoryN<ND> const &t,
                   std::string const      kt,
                   float const            os,
                   Index const            nC = 1,
                   Basis const           &b = IdBasis(),
                   Index const            sgW = 32) -> std::shared_ptr<Grid<ND, VCC>>;
  Grid(
    TrajectoryN<ND> const &traj, std::string const ktype, float const osamp, Index const nC, Basis const &b, Index const sgW);
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
};

} // namespace TOps
} // namespace rl
