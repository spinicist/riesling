#pragma once

#include "op/recon.hpp"

namespace rl {

auto Scaling(std::string const &type, Ops::Op<Cx>::Ptr const A, Ops::Op<Cx>::Ptr const P, Ops::Op<Cx>::CMap const b) -> float;

} // namespace rl
