#pragma once

#include "basis/basis.hpp"
#include "grid-mapping.hpp"
#include "kernel/kernel.hpp"
#include "parse_args.hpp"
#include "top.hpp"

namespace rl {

struct GridOpts
{
  GridOpts(args::Subparser &parser);
  args::ValueFlag<std::string> ktype;
  args::ValueFlag<float>       osamp;
  args::Flag                   vcc;
  args::ValueFlag<Index>       batches, subgridSize, splitSize;
};

namespace TOps {
template <int ND, bool VCC = false> struct Grid final : TOp<Cx, ND + 2 + VCC, 3>
{
  TOP_INHERIT(Cx, ND + 2 + VCC, 3)
  using Parent::adjoint;
  using Parent::forward;
  std::shared_ptr<Kernel<Scalar, ND>> kernel;
  Mapping<ND>                         mapping;
  Basis<Scalar>                       basis;
  std::optional<Mapping<ND>>          vccMapping;

  static auto Make(TrajectoryN<ND> const &t,
                   std::string const      kt,
                   float const            os,
                   Index const            nC = 1,
                   Basis<Scalar> const   &b = IdBasis<Scalar>(),
                   Index const            bSz = 32,
                   Index const            sSz = 16384) -> std::shared_ptr<Grid<ND, VCC>>;
  Grid(TrajectoryN<ND> const &traj,
       std::string const      ktype,
       float const            osamp,
       Index const            nC,
       Basis<Scalar> const   &b,
       Index const            bSz,
       Index const            sSz);
  Grid(std::shared_ptr<Kernel<Scalar, ND>> const &k,
       Mapping<ND> const                          m,
       Index const                                nC,
       Basis<Scalar> const                       &b = IdBasis<Scalar>());
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;
};

} // namespace TOps
} // namespace rl
