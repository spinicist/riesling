#pragma once

#include "parse_args.hpp"
#include "trajectory.hpp"
#include "types.hpp"

namespace rl {

struct ROVIROpts
{
  ROVIROpts(args::Subparser &parser);
  args::ValueFlag<Eigen::Array3f, Array3fReader> res;
  args::ValueFlag<float> fov;
  args::ValueFlag<Index> gap;
};

auto ROVIR(
  ROVIROpts &opts, Trajectory const &traj, float const energy, Index const channels, Index const lorestraces, Cx4 const &data)
  -> Eigen::MatrixXcf;

} // namespace rl
