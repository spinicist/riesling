#pragma once

#include "log.h"
#include "op/sdc.hpp"
#include "parse_args.h"
#include "trajectory.h"
#include "types.h"

namespace rl {

namespace SDC {
struct Opts
{
  Opts(args::Subparser &parser);
  args::ValueFlag<std::string> type;
  args::ValueFlag<float> pow;
};

R2 Pipe(Trajectory const &traj, bool const nn, float const os, Index const max_its = 40);
R2 Radial(Trajectory const &traj, Index const lores, Index const gap);
std::unique_ptr<SDCOp> Choose(Opts &opts, Trajectory const &t, float const os);

} // namespace SDC
} // namespace rl
