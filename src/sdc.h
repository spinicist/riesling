#pragma once

#include "log.hpp"
#include "op/sdc.hpp"
#include "parse_args.hpp"
#include "trajectory.hpp"
#include "types.h"

namespace rl {

namespace SDC {
struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string> type;
  args::ValueFlag<float> pow;
  args::ValueFlag<Index> maxIterations;
};

Re2 Pipe(Trajectory const &traj, std::string const &ktype, float const os, Index const max_its = 40);
Re2 Radial(Trajectory const &traj, Index const lores, Index const gap);
std::unique_ptr<SDCOp> Choose(Opts &opts, Trajectory const &t, std::string const &ktype, float const os);

} // namespace SDC
} // namespace rl
