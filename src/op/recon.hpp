#pragma once

#include "nufft.hpp"
#include "sense.hpp"
#include "multiply.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "../sense.hpp"
#include "io/reader.hpp"

namespace rl {

using ReconOp = MultiplyOp<SenseOp, Operator<Cx, 5, 4>>;

ReconOp Recon(
  CoreOpts &coreOpts,
  SDC::Opts &sdcOpts,
  SENSE::Opts &senseOpts,
  Trajectory const &traj,
  bool const toeplitz,
  HD5::Reader &reader);

} // namespace rl
