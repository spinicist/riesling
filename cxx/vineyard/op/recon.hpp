#pragma once

#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "sense/sense.hpp"
#include "sys/args.hpp"

/*
 *  Full recon operators
 */

namespace rl {
namespace Recon {

auto Single(GridOpts &gridOpts, Trajectory const &traj, Index const nSlab, Index const nTime, Basis::CPtr basis)
  -> TOps::TOp<Cx, 5, 5>::Ptr;

auto SENSE(GridOpts         &gridOpts,
           SENSE::Opts      &senseOpts,
           Trajectory const &traj,
           Index const       nSlab,
           Index const       nTime,
           Basis::CPtr       basis,
           Cx5 const        &data) -> TOps::TOp<Cx, 5, 5>::Ptr;

} // namespace Recon
} // namespace rl