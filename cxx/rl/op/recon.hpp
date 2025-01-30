#pragma once

#include "../io/reader.hpp"
#include "../op/compose.hpp"
#include "../op/nufft.hpp"
#include "../precon.hpp"
#include "../sense/sense.hpp"

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

  struct f0Opts
  {
    std::string        fname;
    std::vector<float> τ;
  };

  Recon(Opts const                &rOpts,
        PreconOpts const          &pOpts,
        TOps::Grid<3>::Opts const &gridOpts,
        SENSE::Opts const         &senseOpts,
        Trajectory const          &traj,
        Basis::CPtr                basis,
        f0Opts const              &f0Opts,
        Cx5 const                 &data);
  TOps::TOp<Cx, 5, 5>::Ptr A, M;
};
} // namespace rl