#pragma once

#include "op/recon.hpp"
#include "parse_args.hpp"

namespace rl {

auto Scaling(args::ValueFlag<std::string> &type, std::shared_ptr<ReconOp> const &recon, Cx5 const &data) -> float;

} // namespace rl
