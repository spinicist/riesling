#pragma once

#include "op/recon.hpp"
#include "parse_args.hpp"

namespace rl {

auto Scaling(
  args::ValueFlag<std::string> &type, Ops::Op<Cx>::Ptr const A, Ops::Op<Cx>::Ptr const P, Cx *const b)
  -> float;

} // namespace rl
