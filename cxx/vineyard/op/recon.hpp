#pragma once

#include "args.hpp"
#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "sense/sense.hpp"

/*
 *  Full recon operators
 */

namespace rl {
namespace Recon {

auto SENSE(bool const        ndft,
           GridOpts         &gridOpts,
           SENSE::Opts      &senseOpts,
           Trajectory const &traj,
           Index const       nSlab,
           Index const       nTime,
           Basis::CPtr       basis,
           Cx5 const        &data) -> TOps::TOp<Cx, 5, 5>::Ptr;

auto Channels(bool const        ndft,
              GridOpts         &gridOpts,
              Trajectory const &traj,
              Index const       nC,
              Index const       nSlab,
              Index const       nTime,
              Basis::CPtr       basis,
              Sz3 const         shape = Sz3()) -> TOps::TOp<Cx, 6, 5>::Ptr;

} // namespace Recon
} // namespace rl