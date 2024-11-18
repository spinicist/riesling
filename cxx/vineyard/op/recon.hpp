#pragma once

#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "precon.hpp"
#include "sense/sense.hpp"
#include "sys/args.hpp"

/*
 *  Full recon operators
 */

namespace rl {
struct Recon
{
  struct Opts
  {
    bool decant, lowmem;
  };

  Recon(Opts const                &rOpts,
        PreconOpts const          &pOpts,
        TOps::Grid<3>::Opts const &gridOpts,
        SENSE::Opts               &senseOpts,
        Trajectory const          &traj,
        Basis::CPtr                basis,
        Cx5 const                 &data);
  TOps::TOp<Cx, 5, 5>::Ptr A, M;
};
} // namespace rl