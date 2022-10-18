#pragma once

#include "func/multiply.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "trajectory.hpp"
#include "types.hpp"

namespace rl {

namespace SDC {

template <int ND>
Re2 Pipe(Trajectory const &traj, std::string const &ktype, float const os, Index const max_its = 40);
Re2 Radial(Trajectory const &traj, Index const lores, Index const gap);

struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string> type;
  args::ValueFlag<float> pow;
  args::ValueFlag<Index> maxIterations;
};

using Functor = BroadcastMultiply<Cx, 3>;

auto Choose(Opts &opts, Trajectory const &t, Index const nC, std::string const &ktype, float const os)
  -> std::optional<Functor>;

} // namespace SDC
} // namespace rl
