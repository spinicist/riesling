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

auto SENSERecon(CoreOpts                       &coreOpts,
                GridOpts                       &gridOpts,
                Trajectory const               &traj,
                Index const                     nSlab,
                std::shared_ptr<SenseOp> const &sense,
                Basis<Cx> const                &basis) -> TOp<Cx, 4, 4>::Ptr;

auto Channels(CoreOpts         &coreOpts,
              GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Basis<Cx> const  &basis) -> TOp<Cx, 5, 4>::Ptr;

} // namespace rl
