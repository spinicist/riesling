#pragma once

#include "../io/reader.hpp"
#include "../op/compose.hpp"
#include "../precon.hpp"
#include "../sense/sense.hpp"
#include "recon-opts.hpp"

/*
 *  Full recon operators
 */

namespace rl {
struct Recon2
{
  Recon2(ReconOpts const      &rOpts,
         PreconOpts const     &pOpts,
         GridOpts<2> const    &gridOpts,
         SENSE::Opts const    &senseOpts,
         TrajectoryN<2> const &traj,
         Basis::CPtr           basis,
         Cx5 const            &data);

  Recon2(ReconOpts const      &rOpts,
         PreconOpts const     &pOpts,
         GridOpts<2> const    &gridOpts,
         SENSE::Opts const    &senseOpts,
         TrajectoryN<2> const &traj,
         f0Opts const         &f0Opts,
         Cx5 const            &data,
         Re3 const            &f0map);
  TOps::TOp<Cx, 5, 5>::Ptr A, M;
};
} // namespace rl
