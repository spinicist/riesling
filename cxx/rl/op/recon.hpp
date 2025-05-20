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
struct Recon
{
  Recon(ReconOpts const   &rOpts,
        PreconOpts const  &pOpts,
        GridOpts<3> const &gridOpts,
        SENSE::Opts const &senseOpts,
        Trajectory const  &traj,
        Basis::CPtr        basis,
        Cx5 const         &data);

  Recon(ReconOpts const   &rOpts,
        PreconOpts const  &pOpts,
        GridOpts<3> const &gridOpts,
        SENSE::Opts const &senseOpts,
        Trajectory const  &traj,
        f0Opts const      &f0Opts,
        Cx5 const         &data,
        Re3 const         &f0map);
  TOps::TOp<Cx, 5, 5>::Ptr A, M;
};
} // namespace rl