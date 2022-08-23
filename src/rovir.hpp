#pragma once

#include "parse_args.h"
#include "trajectory.h"
#include "types.h"

namespace rl {

struct ROVIROpts
{
  ROVIROpts(args::Subparser &parser);
  args::ValueFlag<float> res;
  args::ValueFlag<float> fov;
  args::ValueFlag<float> loThresh;
  args::ValueFlag<float> hiThresh;
  args::ValueFlag<float> gap;
};

auto ROVIR(
  ROVIROpts &opts,
  Trajectory const &traj,
  float const energy,
  Index const channels,
  Index const loresSpokes,
  Cx3 const &data) -> Eigen::MatrixXcf;

} // namespace rl
