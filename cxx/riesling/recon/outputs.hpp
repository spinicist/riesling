#pragma once

#include "info.hpp"
#include "io/hd5-core.hpp"
#include "op/grid.hpp"
#include "op/top.hpp"
#include "precon.hpp"
#include "sense/sense.hpp"
#include "types.hpp"

namespace rl {
template <int ND>
void WriteOutput(
  std::string const &cmd, std::string const &fname, CxN<ND> const &img, HD5::DimensionNames<ND> const &dims, Info const &info);

void WriteResidual(std::string const              &cmd,
                   std::string const              &writer,
                   GridOpts                       &gridOpts,
                   SENSE::Opts                    &senseOpts,
                   PreconOpts                     &preOpts,
                   Trajectory const               &traj,
                   Cx5CMap const                  &x,
                   TOps::TOp<Cx, 5, 5>::Ptr const &A,
                   Cx5                            &noncart);
} // namespace rl