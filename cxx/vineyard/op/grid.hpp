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

struct GridOpts
{
  GridOpts(args::Subparser &parser);
  args::ValueFlag<Eigen::Array3f, Array3fReader> fov;
  args::ValueFlag<rl::Sz3, SzReader<3>>          matrix;
  args::ValueFlag<float>                         osamp;
  args::ValueFlag<std::string>                   ktype;
  args::Flag                                     vcc;
  args::ValueFlag<Index>                         batches, subgridSize;
};

namespace TOps {
template <int ND, bool VCC = false> struct Grid final : TOp<Cx, ND + 2 + VCC, 3>
{
  TOP_INHERIT(Cx, ND + 2 + VCC, 3)
  TOP_DECLARE(Grid)
  static auto
  Make(TrajectoryN<ND> const &t, float const os, std::string const kt, Index const nC, Basis::CPtr b, Index const sgW = 32)
    -> std::shared_ptr<Grid<ND, VCC>>;
  Grid(TrajectoryN<ND> const &traj, float const osamp, std::string const ktype, Index const nC, Basis::CPtr b, Index const sgW);
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

  std::shared_ptr<KernelBase<Scalar, ND>> kernel;

private:
  Index                    subgridW;
  std::vector<Mapping<ND>> mappings;
  std::vector<std::mutex> mutable mutexes;
  Basis::CPtr                             basis;
  std::optional<std::vector<Mapping<ND>>> vccMapping;
};

} // namespace TOps
} // namespace rl
