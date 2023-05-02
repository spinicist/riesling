#pragma once

#include "io/reader.hpp"
#include "op/compose.hpp"
#include "op/nufft.hpp"
#include "op/sense.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

namespace rl {

using ReconOp = Compose<SenseOp, TensorOperator<Cx, 5, 4>>;

auto make_recon(
  CoreOpts &coreOpts,
  SDC::Opts &sdcOpts,
  SENSE::Opts &senseOpts,
  Trajectory const &traj,
  HD5::Reader &reader) -> std::shared_ptr<ReconOp>;

} // namespace rl
