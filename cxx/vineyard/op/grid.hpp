#pragma once

#include "basis/basis.hpp"
#include "grid-mapping.hpp"
#include "kernel/kernel.hpp"
#include "sys/args.hpp"
#include "top.hpp"

#include <args.hxx>
#include <mutex>
#include <optional>

namespace rl {

template <int ND> struct GridOpts
{
  ArrayFlag<float, ND>         fov;
  SzFlag<ND>                   matrix;
  args::ValueFlag<float>       osamp;
  args::ValueFlag<std::string> ktype;
  args::Flag                   vcc, lowmem;
  args::ValueFlag<Index>       subgridSize;

  GridOpts(args::Subparser &parser)
    : fov(parser, "FOV", "Grid FoV in mm (x,y,z)", {"fov"}, Eigen::Array<float, ND, 1>::Zero())
    , matrix(parser, "M", "Grid matrix size", {"matrix", 'm'}, Sz<ND>())
    , osamp(parser, "O", "Grid oversampling factor (1.3)", {"osamp"}, 1.3f)
    , ktype(parser, "K", "Grid kernel - NN/KBn/ESn (ES4)", {'k', "kernel"}, "ES4")
    , vcc(parser, "V", "Virtual Conjugate Coils", {"vcc"})
    , lowmem(parser, "L", "Low memory mode", {"lowmem", 'l'})
    , subgridSize(parser, "B", "Subgrid size (8)", {"subgrid-size"}, 8)
  {
  }
};

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
  Index                           subgridW;
  std::vector<SubgridMapping<ND>> subs;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr                                    basis;
  std::optional<std::vector<SubgridMapping<ND>>> vccSubs;
};

} // namespace TOps
} // namespace rl
