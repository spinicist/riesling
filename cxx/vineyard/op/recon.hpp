#pragma once

#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "op/sense.hpp"
#include "parse_args.hpp"

/*
 *  Full recon operators
 */

namespace rl {
namespace Recon {

auto SENSE(CoreOpts               &coreOpts,
           GridOpts               &gridOpts,
           Trajectory const       &traj,
           Index const             nSlab,
           TOps::SENSE::Ptr const &sense,
           Basis<Cx> const        &basis) -> TOps::TOp<Cx, 4, 4>::Ptr;

auto Channels(bool const            ndft,
              GridOpts             &gridOpts,
              Trajectory const     &traj,
              Eigen::Array3f const &fov,
              Index const           nC,
              Index const           nSlab,
              Basis<Cx> const      &basis) -> TOps::TOp<Cx, 5, 4>::Ptr;

} // namespace Recon
} // namespace rl