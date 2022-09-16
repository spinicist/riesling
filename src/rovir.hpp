#pragma once

#include "parse_args.hpp"
#include "trajectory.hpp"
#include "types.hpp"

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
  Index const lorestraces,
  Cx3 const &data) -> Eigen::MatrixXcf;

} // namespace rl
