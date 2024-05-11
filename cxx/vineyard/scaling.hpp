#pragma once

#include "op/recon.hpp"
#include "parse_args.hpp"

namespace rl {

auto Scaling(
  args::ValueFlag<std::string> &type, TOp<Cx, 4, 4>::Ptr const A, std::shared_ptr<Ops::Op<Cx>> const P, Cx *const b)
  -> float;

} // namespace rl
