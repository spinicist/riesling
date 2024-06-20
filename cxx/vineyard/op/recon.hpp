#pragma once

#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "sense/sense.hpp"

/*
 *  Full recon operators
 */

namespace rl {
namespace Recon {

auto SENSE(CoreOpts         &coreOpts,
           GridOpts         &gridOpts,
           SENSE::Opts      &senseOpts,
           Trajectory const &traj,
           Index const       nSlab,
           Basis<Cx> const  &basis,
           Cx5 const        &data) -> TOps::TOp<Cx, 4, 4>::Ptr;

auto Channels(bool const        ndft,
              GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Basis<Cx> const  &basis,
              Sz3 const         shape = Sz3()) -> TOps::TOp<Cx, 5, 4>::Ptr;

} // namespace Recon
} // namespace rl