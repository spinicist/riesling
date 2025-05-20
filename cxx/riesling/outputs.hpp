#pragma once

#include "rl/info.hpp"
#include "rl/io/hd5-core.hpp"
#include "rl/op/grid-opts.hpp"
#include "rl/op/recon.hpp"
#include "rl/op/top.hpp"
#include "rl/precon.hpp"
#include "rl/sense/sense.hpp"
#include "rl/types.hpp"

namespace rl {
template <int ND> void WriteOutput(
  std::string const &cmd, std::string const &fname, CxN<ND> const &img, HD5::DimensionNames<ND> const &dims, Info const &info);

void WriteResidual(std::string const              &cmd,
                   std::string const              &writer,
                   ReconOpts const                &reconOpts,
                   GridOpts<3> const              &gridOpts,
                   SENSE::Opts const              &senseOpts,
                   PreconOpts const               &preOpts,
                   Trajectory const               &traj,
                   Cx5CMap const                  &x,
                   TOps::TOp<Cx, 5, 5>::Ptr const &A,
                   Cx5                            &noncart);
} // namespace rl