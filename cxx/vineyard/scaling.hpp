#pragma once

#include "op/recon.hpp"
#include "parse_args.hpp"

namespace rl {

auto Scaling(
  args::ValueFlag<std::string> &type, std::shared_ptr<ReconOp> const A, std::shared_ptr<Ops::Op<Cx>> const P, Cx *const b)
  -> float;

} // namespace rl
