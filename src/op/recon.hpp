#pragma once

#include "../sense.hpp"
#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.hpp"

namespace rl {

using ReconOp = Compose<SenseOp, Operator<Cx, 5, 4>>;

auto make_recon(
  CoreOpts &coreOpts,
  SDC::Opts &sdcOpts,
  SENSE::Opts &senseOpts,
  Trajectory const &traj,
  bool const toeplitz,
  HD5::Reader &reader) -> std::shared_ptr<ReconOp>;

} // namespace rl
